"""
ETCL Utils - Modified utillsv2.py

Changes from original:
- Loads confidence scores along with pseudo labels
- Supports both .npy (C2FPL) and .npz (ETCL) formats
"""

from matplotlib.pyplot import axis
import numpy as np
import torch.utils.data as data
import pandas as pd
import torch
import os

# Import option if available
try:
    import option
    args = option.parser.parse_args()
except:
    args = None


def Concat_list_all_crop_feedback(Test=False, create='False', args=None):
    """
    Load test data or pseudo labels
    
    For ETCL: Also loads confidence scores
    
    Returns:
        If Test=True: test features
        If Test=False: (num_labels, labels_tensor, confidence_tensor)
    """
    from datetime import datetime
    
    if Test is True:
        # Load test data
        test_path = 'concatenated/Concat_test_10.npy'
        if os.path.exists(test_path):
            con_test = np.load(test_path)
        else:
            # Try memmap for large files
            con_test = np.memmap('../C2FPL/Concat_test_10.npy', 
                                dtype='float32', mode='r', 
                                shape=(290, 32, 10, 2048)).copy()
        print('Testset size:', con_test.shape)
        return con_test
    
    if Test is False:
        # Get pseudo label file path
        if args is not None and hasattr(args, 'pseudofile'):
            pseudo_path = args.pseudofile
        else:
            pseudo_path = 'Unsup_labels/etcl_labels.npz'  # Default ETCL path
        
        print(f'Loading Pseudo Labels from: {pseudo_path}')
        
        # Try loading as NPZ (ETCL format with confidence)
        if pseudo_path.endswith('.npz') and os.path.exists(pseudo_path):
            data = np.load(pseudo_path)
            labels = data['labels']
            confidence = data['confidence']
            print(f'[*] Loaded ETCL labels: {labels.shape}, confidence: {confidence.shape}')
            
            return (
                len(labels),
                torch.tensor(labels).cuda(),
                torch.tensor(confidence).cuda()
            )
        
        # Try loading as NPY (C2FPL format, no confidence)
        elif pseudo_path.endswith('.npy') and os.path.exists(pseudo_path):
            labels = np.load(pseudo_path)
            print(f'[*] Loaded C2FPL labels: {labels.shape}')
            
            # Check if separate confidence file exists
            conf_path = pseudo_path.replace('.npy', '_confidence.npy')
            conf_path2 = pseudo_path.replace('_labels.npy', '_confidence.npy')
            
            if os.path.exists(conf_path):
                confidence = np.load(conf_path)
                print(f'[*] Loaded confidence: {confidence.shape}')
            elif os.path.exists(conf_path2):
                confidence = np.load(conf_path2)
                print(f'[*] Loaded confidence: {confidence.shape}')
            else:
                # No confidence file - use uniform confidence
                confidence = np.ones_like(labels)
                print('[*] No confidence file found, using uniform confidence')
            
            return (
                len(labels),
                torch.tensor(labels).cuda(),
                torch.tensor(confidence).cuda()
            )
        
        # Try NPZ without extension
        elif os.path.exists(pseudo_path + '.npz'):
            data = np.load(pseudo_path + '.npz')
            labels = data['labels']
            confidence = data['confidence']
            return (
                len(labels),
                torch.tensor(labels).cuda(),
                torch.tensor(confidence).cuda()
            )
        
        else:
            raise FileNotFoundError(f"Pseudo label file not found: {pseudo_path}")


def Concat_list_all_crop_feedback_legacy(Test=False, create='False'):
    """
    Original C2FPL version (for backward compatibility)
    """
    from datetime import datetime

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    
    if Test is True:
        con_test = np.memmap('../C2FPL/Concat_test_10.npy', 
                            dtype='float32', mode='r', 
                            shape=(290, 32, 10, 2048)).copy()
        print('Testset size:', con_test.shape)
        return con_test
    
    if Test is False:
        if create == 'True':
            print('loading Pseudo Labels......', args.pseudofile)
        label_all = np.load(args.pseudofile)
        print('[*] concatenated labels shape:', label_all.shape)
        return len(label_all), torch.tensor(label_all).cuda()


# Helper function for video-level operations
def load_video_segments(features_path='concatenated/concat_UCF.npy', 
                        nalist_path='nalist.npy'):
    """
    Load features and split by video
    
    Returns:
        videos: list of video features
        nalist: video index ranges
    """
    train_data = np.load(features_path)
    nalist = np.load(nalist_path)
    
    videos = []
    for fromid, toid in nalist:
        videos.append(train_data[fromid:toid])
    
    return videos, nalist
