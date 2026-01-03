import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from dataset import UCFTestVideoDataset
from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import time
import os
from model import Model, Model_V2
# from datasets.dataset import 
'''
def test(dataloader, model, args, device):
    with torch.no_grad():#  #
        model.eval()
        
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            # print(input.size())
            startime = time.time()
            logits = model(inputs=input)
            # print("done in {0}.".format(time.time() - startime))
            pred = torch.cat((pred, logits))
            if i == 0:
                print("input:", input.shape)
                print("logits:", logits.shape)   #(B,1,2048) 


            
        gt = np.load(args.gt, allow_pickle=True)
        # print(gt.shape)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        #gt = gt[:len(pred)] 

        print(f"DEBUG: len(gt) = {len(list(gt))}")
        print(f"DEBUG: len(pred) = {len(pred)}")

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc: ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)

        # np.save('UCF_pred/'+'{}-pred_UCFV1_i3d.npy'.format(epoch), pred)
        return rec_auc, pr_auc
    '''
def test(dataloader, model, args, device):
    model.eval()
    gt_all = np.load(args.gt, allow_pickle=True)
    print("gt_all shape:", gt_all.shape, "dtype:", gt_all.dtype)
    assert gt_all.ndim == 1, f"gt_all must be 1D frame array, got shape {gt_all.shape}, dtype={gt_all.dtype}"

    ptr = 0
    total_frames = len(gt_all)

    preds = []
    gts = []

    with torch.no_grad():
        for i, x in enumerate(dataloader):
            # x: (1,T,2048)
            x = x.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            logits = model(inputs=x)                 # (1,T,1)
            logits = logits.squeeze(0).squeeze(-1)   # (T,)

            prob = torch.sigmoid(logits)
            pred = prob.detach().cpu().numpy()

            T = pred.shape[0]
            pred_frame = np.repeat(pred, 16)         # (T*16,)

            need = T * 16
            # 슬라이스 전에 범위 초과 체크 (가장 중요)
            assert ptr + need <= total_frames, (
                f"GT slice out of range at video {i}: ptr={ptr}, need={need}, total={total_frames}"
            )

            gt_i = gt_all[ptr:ptr + T*16]
            ptr += T*16

            preds.append(pred_frame)
            gts.append(gt_i)

            if i == 0:
                if not hasattr(test, "_printed_first_video"):
                    print("test video0 T:", T, "pred_frame:", pred_frame.shape, "gt_i:", gt_i.shape)
                    test._printed_first_video = True

    pred_all = np.concatenate(preds)
    gt_all2 = np.concatenate(gts)

    if not hasattr(test, "_printed_len"):
        print("DEBUG: len(gt) =", len(gt_all2))
        print("DEBUG: len(pred) =", len(pred_all))
        test._printed_len = True
        
    assert len(pred_all) == len(gt_all2), (len(pred_all), len(gt_all2))

    fpr, tpr, _ = roc_curve(gt_all2, pred_all)
    #np.save('fpr.npy', fpr)
    #np.save('tpr.npy', tpr)
    rec_auc = auc(fpr, tpr)

    print('auc: ' + str(rec_auc))

    precision, recall, _ = precision_recall_curve(gt_all2, pred_all)
    pr_auc = auc(recall, precision)

    assert ptr == total_frames, (ptr, total_frames)
    print("ptr ok:", ptr)

    np.save('precision.npy', precision)
    np.save('recall.npy', recall)


    # np.save('UCF_pred/'+'{}-pred_UCFV1_i3d.npy'.format(epoch), pred)
    return rec_auc, pr_auc



def test_2(dataloader, model, args, device):
    with torch.no_grad():  
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)

            logits = model(inputs=input)

            pred = torch.cat((pred, logits))


            
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # gt = gt[:len(pred)] 

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        np.save('fpr1.npy', fpr)
        np.save('tpr1.npy', tpr)
        rec_auc = auc(fpr, tpr)
        # print('auc: ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print("AP = ",pr_auc)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        np.save('pred_XD.npy', pred)
        
        return rec_auc, pr_auc
    

if __name__ == '__main__':
    args = option.parser.parse_args()
    #gt = np.load(args.gt)
    # con_all = np.load('{}.npy'.format(args.conall))
    device = torch.device("cuda")
    model = Model_V2(args.feature_size).to(device)
    test_loader = DataLoader(UCFTestVideoDataset(conall_path="../C2FPL/Concat_test_10.npy",
                            nalist_path="list/nalist_test_i3d.npy"), 
                            batch_size=1, shuffle=False, 
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    import glob
    pattern = os.path.join("unsupervised_ckpt", f"{args.datasetname}_best_*.pkl")
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise FileNotFoundError(f"no ckpt: {pattern}")

    ckpt_path = max(ckpts, key = os.path.getmtime)
    print("Loading:", ckpt_path)
    
    state = torch.load(ckpt_path, map_location=device)

    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    auc, ap = test(test_loader, model, args, device)
    print("AUC:", auc, "AP:", ap)
    