import torch.utils.data as data
import numpy as np
from utillsv2 import  Concat_list_all_crop_feedback
import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
from tqdm import tqdm
import option

args = option.parser.parse_args()

class Dataset_Con_all_feedback_UCF(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        #변경(추가) -- modality 인자 안 받고 있음.
        if not hasattr(args, "modality"):
            args.modality = "rgb"
        self.modality = args.modality
        self.is_normal = is_normal

        if test_mode:
            self.con_all = Concat_list_all_crop_feedback(True)

        else:
            #변경 -- memmap (shape은 nalist 활용)
            # self.con_all = np.load("concatenated/concat_UCF.npy")
            # self.con_all = np.load("concatenated/concat_XD.npy")
            nalist = np.load('../../C2FPL/list/nalist_i3d.npy')
            total_T = int(nalist[-1,1])
            self.con_all = np.memmap('../../C2FPL/concat_UCF.npy', dtype='float32', mode='r', shape=(total_T, 10, 2048)) 

            print('self.con_all shape:',self.con_all.shape)

        self.tranform = transform
        self.test_mode = test_mode 

    def __getitem__(self, index):        
        if self.test_mode:
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)

        else:    
            features = self.con_all[index]
            features = np.array(features, dtype=np.float32)
    
        if self.test_mode:
            return features
        else:
            return features , index

    def __len__(self):
        return len(self.con_all)   



class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.transform = transform
        self.test_mode = test_mode
        nalist = np.load('list/nalist_XD_test_R50NL.npy')
        self.total_T = int(nalist[-1,1])


        if test_mode:
            # XD test feature: shape = (145649, 1024)
            self.con_all = np.memmap(args.xd_feat, dtype="float32", mode="r", shape=(self.total_T, 10, 2048))
            print('[XD test] self.con_all shape:', self.con_all.shape)

            #assert self.con_all.ndim == 2, f"Expected 2D array, got {self.con_all.shape}"
            #assert self.con_all.shape[1] == 1024, f"Expected feature dim 1024, got {self.con_all.shape[1]}"

        else:
            raise NotImplementedError("지금은 XD test_mode=True만.")

    def __getitem__(self, index):
        features = self.con_all[index]
        features = np.array(features, dtype=np.float32)   # (1024,)

        if self.test_mode:
            return features
        else:
            return features, index

    def __len__(self):
        return len(self.con_all)