from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utillsv2 import Concat_list_all_crop_feedback
from model import Model, Model_V2
from dataset import  UCFTestVideoDataset, UCFTrainSnippetDataset
from train import concatenated_train, concatenated_train_feedback
from test import test
import option
from tqdm import tqdm

import os
import numpy as np
import wandb
import copy
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print('mainv2')
    args = option.parser.parse_args()
    set_seed(42)

    len_N, original_lables  = Concat_list_all_crop_feedback(Test=False, create='False')

    wandb.login()
    wandb.init(project="Unsupervised Anomaly Detection", config=args)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = wandb.run.id

    best_path  = f'unsupervised_ckpt/{args.datasetname}_best_{ts}_{run_id}.pkl'
    final_path = f'unsupervised_ckpt/{args.datasetname}_final_{ts}_{run_id}.pkl'

    test_loader = DataLoader(UCFTestVideoDataset("Concat_test_10.npy", "list/nalist_test_i3d.npy"), 
                            batch_size=1, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    
    train_loader = DataLoader(UCFTrainSnippetDataset("concat_UCF.npy", args.pseudofile), 
                                batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    
    model = Model_V2(args.feature_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    
    auc, ap = test(test_loader, model, args, device)
    
    print("epcoh 0 auc = ", auc)
    wandb.log({'AUC': auc,'AP': ap}, step=0)
    best_auc = auc

    #epoch 0 모델도 best로 저장함
    torch.save(model.state_dict(), best_path)
    print("init best_auc:", best_auc, "->", best_path)

    test_info = {"epoch": [], "test_auc": []}

    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        loss, lls = concatenated_train_feedback(train_loader, model, optimizer,original_lables, device )
        auc, ap = test(test_loader, model, args, device)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] epoch {epoch} best_auc updated: {best_auc:.4f}-> saved to {best_path}")
        test_info["epoch"].append(epoch)
        test_info["test_auc"].append(auc)
        scheduler.step()
        
        update = sorted(lls, key=lambda x: x[0])

        print('\nEpoch {}/{}, LR: {:.4f} auc: {:.4f}, ap: {:.4f}, loss: {:.4f}\n'.format(epoch, args.max_epoch, optimizer.param_groups[0]['lr'] , auc, ap, loss))
        wandb.log({'AUC': auc,'AP': ap, 'loss': loss}, step=(epoch+1)*544)

    #wandb.run.name = args.datasetname
    torch.save(model.state_dict(),final_path)
    print("saved final ->", final_path)


        
