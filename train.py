import numpy as np
import torch
import torch.nn.functional as F
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import option
import math 
from tqdm import tqdm

args = option.parser.parse_args()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCELoss(reduction='none')
# loss_fn = torch.nn.HingeEmbeddingLoss()


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def _init_(self):
        super(SigmoidMAELoss, self)._init_()
        from torch.nn import Sigmoid
        self._sigmoid_ = Sigmoid()
        self._l1_loss_ = MSELoss()

    def forward(self, pred, target):
        return self._l1_loss_(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def _init_(self):
        super(SigmoidCrossEntropyLoss, self)._init_()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))







def concatenated_train(loader, model, optimizer,device):
    with torch.set_grad_enabled(True):
        model.train()

        losses = []
        # for _ in range(30):  # 800/batch_size
        for _, (input, labels) in enumerate(loader):
            # input, labels = next(iter(loader)) 
            input, labels = input.to(device), labels.to(device)
            labels = labels.float()

            
            scores = model(input)
            scores = scores.float().flatten()
            loss = loss_fn(scores, labels)
            losses.append(loss.cpu().detach().numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        return np.mean(losses)




def concatenated_train_top(loader, model, optimizer,device):
    with torch.set_grad_enabled(True):
        model.train()
        i= 0
        losses = []
        # for _ in range(30):  # 800/batch_size
        # for _ in range(100):
        for _, (input, labels) in enumerate(loader):
            
            # input, labels = next(iter(loader)) 
            input, labels = input.to(device), labels.to(device)
            labels = labels.float()
            # print(input.size())
            scores = model(input)
            # scores, feat_select_top, feat_select_low, top_select_score = model(input)
            scores = scores.squeeze()
            # print(scores.size())

            loss_criterion = loss_fn(scores, labels)

            # loss_sparse = sparsity(feat_select_top, 128, 8e-2) 
            # loss_smooth = smooth(top_select_score, 8e-3)

            loss = loss_criterion  #+ loss_sparse #+ loss_smooth #+loss_random(feat_select_low, feat_select_top) +los_clss(feat_select_low, feat_select_top)#+ 
            losses.append(loss.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            # print(i)

        return np.mean(losses)

#schedule function (basic)
def get_gamma(epoch, warmup_epochs=10, gamma_start=0.95, gamma_end=0.60):
    if epoch <= warmup_epochs:
        t = (epoch - 1) / max(1, warmup_epochs - 1)
        return gamma_start + (gamma_end - gamma_start) * t
    return gamma_end


def concatenated_train_feedback(loader, model, optimizer, hard_all, conf_all, device, epoch):
    with torch.set_grad_enabled(True):

        model.train()

        losses = []
        #original_labels = original_label
        new_labels = []

        import time
        start = time.time()

        #gamma = get_gamma(epoch, warmup_epochs=10, gamma_start=0.8, gamma_end=0.4)

        for it, (input, idx) in enumerate(loader):
            if it==0:
                print(">>> got first batch", input.shape, idx.shape)
            #먼저 GPU로 올리기
            input = input.to(device, non_blocking = True)

            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long)
            idx = idx.to(device, non_blocking=True)

            hard = hard_all[idx]
            conf = conf_all[idx]

            #mask = (hard != -1) & (conf >= gamma) #첫번째 시도 - threshold 기준으로 자르기

            #if mask.sum().item() == 0:
                #continue

            base = (hard != -1)   #변경, 두번째 시도 -> top-k 기준으로 자르기
            if base.sum().item() == 0:
                continue

            #conf_base = conf[base] 
            p0, p1, warm = 0.05, 0.50, 10 #10epoch 동안 5% ->50%
            t = min(1.0, (epoch-1) / max(1, warm-1))
            p = p0 + (p1-p0) * t
            
            #k = max(1, int(p*conf_base.numel()))
            #thr = torch.topk(conf_base, k, largest=True).values.min()

            #mask = base & (conf >= thr)
            mask = torch.zeros_like(base, dtype=torch.bool)

            # abnormal(1): top-k (conf 큰 것)
            pos = base & (hard == 1)
            k_pos = 0
            if pos.any():
                cpos = conf[pos]
                k_pos = max(4, int(p * cpos.numel())) #최소 4개는 선택되도록 변경
                k_pos = min(k_pos, cpos.numel())
                thr = torch.topk(cpos, k_pos, largest=True).values.min()
                mask[pos] = (conf[pos] >= thr)

            # normal(0): bottom-k (conf 작은 것)  <-- 핵심
            neg = base & (hard == 0)
            if neg.any() and k_pos > 0:
                cneg = conf[neg]
                r = 2
                #k_neg = max(1, int(p * cneg.numel()))
                k_neg = min(int(r*k_pos), cneg.numel()) #pos의 2배개까지로 제한
                thr = torch.topk(cneg, k_neg, largest=False).values.max()  # bottom-k의 최대값
                mask[neg] = (conf[neg] <= thr)

            if mask.sum().item() == 0:
                continue

            #soft = soft.to(device, non_blocking = True)

            #그 다음 gpu tensor로 라벨 인덱싱
            #labels = original_labels[idx].float()
            #input, labels = input.to(device), labels.to(device)
            #labels = labels.float()

            optimizer.zero_grad()
            # scores, feat_select_top, feat_select_low, top_select_score = model(input)
            scores = model(input)
            scores = scores.flatten() #scores = sigmoid 확률
            
            scores_sel = scores[mask]
            labels_sel = hard[mask].float()

            loss_vec = loss_fn(scores_sel, labels_sel)

            conf_sel = conf[mask].detach()
            w = torch.where(labels_sel > 0.5, conf_sel, 1.0 - conf_sel)
            w = w.clamp_min(0.05)

            total_loss = (w * loss_vec).sum() / (w.sum() + 1e-12)

            #loss = loss_fn(scores, soft)
            # loss_sparse = sparsity(scores, 8e-3)
            # loss_smooth = smooth(scores, 8e-4)
            #total_loss = loss # + loss_sparse + loss_smooth
            
            losses.append(total_loss.cpu().detach().item())
            #trans = scores
            # trans = torch.where(scores > 0.6, 1.0, 0.0)
            # trans = (scores + labels)/2
            #trans = trans.cpu().detach().numpy()
            #res = list(zip(soft.cpu().numpy(), trans))

            res = list(zip(idx[mask].detach().cpu().numpy(), scores_sel.detach().cpu().numpy()))
            new_labels += res

            total_loss.backward()
            optimizer.step()

            if it == 0:
                print("first iter done", time.time() - start)
                #print("gamma: ", gamma)
                #print("hard unique:", torch.unique(hard).detach().cpu().numpy())
                base = (hard != -1)
                print("base(hard!=-1):", base.sum().item(), "/", hard.numel())
                print("conf max in batch:", conf.max().item())
                print("selected:", mask.sum().item(), "/", mask.numel())
                c = conf.detach().cpu()
                print("conf percentiles:",
                    "p50", torch.quantile(c, 0.50).item(),
                    "p80", torch.quantile(c, 0.80).item(),
                    "p90", torch.quantile(c, 0.90).item(),
                    "p95", torch.quantile(c, 0.95).item(),
                    "p99", torch.quantile(c, 0.99).item())
                print("batch hard counts:",
                    "normal(0)=", (hard==0).sum().item(),
                    "abnormal(1)=", (hard==1).sum().item(),
                    "ign(-1)=", (hard==-1).sum().item())
                print("selected hard counts:",
                    "0=", (hard[mask]==0).sum().item(),
                    "1=", (hard[mask]==1).sum().item())
            #if it == 10:
                #print("iter 10 done", time.time() - start)
                #break

        #print("epoch time:", time.time() - start)

        return float(np.mean(losses)) if losses else 0.0, new_labels



   