# feature_bridge/detector_runner.py
#feature 를 detector에 넣어서 score/logit 뽑는 역할
import numpy as np
import torch


def run_detector_on_feature(feat_tensor: torch.Tensor, model):
    """
    feat_tensor:
      - (T, 10, 2048) or (T, 2048)

    return:
      prob_np: (T,)
      logit_np: (T,)
    """
    model.eval()
    with torch.no_grad():
        prob, logit = model(feat_tensor, return_logits=True)

    prob = prob.squeeze(-1).detach().cpu().numpy()
    logit = logit.squeeze(-1).detach().cpu().numpy()

    return prob, logit


def summarize_scores(prob: np.ndarray, logit: np.ndarray, name: str = "scores"):
    print(f"[{name}]")
    print(f"  prob shape={prob.shape}, logit shape={logit.shape}")
    print(f"  prob mean={prob.mean():.6f}, std={prob.std():.6f}, max={prob.max():.6f}")
    print(f"  logit mean={logit.mean():.6f}, std={logit.std():.6f}, max={logit.max():.6f}")