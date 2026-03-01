import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import Model_V2
def summarize_feat(feat, name="feat"):
    # feat: (T, 10, 2048) or (T, 2048)
    x = feat.astype(np.float32)

    if x.ndim == 3:
        x_mean_crop = x.mean(axis=1)   # (T, 2048)
    else:
        x_mean_crop = x

    norms = np.linalg.norm(x_mean_crop, axis=-1)

    print(f"[{name}] shape={x.shape}")
    print(f"  mean={x_mean_crop.mean():.6f}, std={x_mean_crop.std():.6f}")
    print(f"  min={x_mean_crop.min():.6f}, max={x_mean_crop.max():.6f}")
    print(f"  L2 norm mean={norms.mean():.6f}, std={norms.std():.6f}")

def compare_feat_cosine(old_feat, new_feat):
    # old_feat, new_feat: (T,10,2048) or (T,2048)
    a = old_feat.mean(axis=1) if old_feat.ndim == 3 else old_feat   # (T,2048)
    b = new_feat.mean(axis=1) if new_feat.ndim == 3 else new_feat

    T = min(len(a), len(b))
    a = a[:T]
    b = b[:T]

    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos = (a_n * b_n).sum(axis=1)

    print(f"[cosine] T={T}")
    print(f"  mean={cos.mean():.6f}, std={cos.std():.6f}")
    print(f"  min={cos.min():.6f}, max={cos.max():.6f}")
    return cos

def compare_detector_scores(old_feat, new_feat, model, device, topk=5):
    """
    old_feat, new_feat:
      shape = (T, 10, 2048) or (T, 2048)

    returns:
      old_prob, new_prob, correlation
    """

    def _to_prob(feat_np):
        x = torch.from_numpy(feat_np).float().to(device)
        model.eval()
        with torch.no_grad():
            prob, logit = model(x, return_logits=True)
        prob = prob.squeeze(-1).detach().cpu().numpy()
        logit = logit.squeeze(-1).detach().cpu().numpy()
        return prob, logit

    old_prob, old_logit = _to_prob(old_feat)
    new_prob, new_logit = _to_prob(new_feat)

    T = min(len(old_prob), len(new_prob))
    old_prob = old_prob[:T]
    new_prob = new_prob[:T]
    old_logit = old_logit[:T]
    new_logit = new_logit[:T]

    corr = np.corrcoef(old_prob, new_prob)[0, 1]

    print(f"[detector score compare] T={T}")
    print(f" old prob mean={old_prob.mean():.6f}, std={old_prob.std():.6f}, max={old_prob.max():.6f}")
    print(f" new prob mean={new_prob.mean():.6f}, std={new_prob.std():.6f}, max={new_prob.max():.6f}")
    print(f" score correlation={corr:.6f}")

    old_top_idx = np.argsort(old_prob)[-topk:][::-1]
    new_top_idx = np.argsort(new_prob)[-topk:][::-1]

    print(f" old top-{topk} idx: {old_top_idx}")
    print(f" old top-{topk} val: {np.round(old_prob[old_top_idx], 4)}")
    print(f" new top-{topk} idx: {new_top_idx}")
    print(f" new top-{topk} val: {np.round(new_prob[new_top_idx], 4)}")

    overlap = len(set(old_top_idx.tolist()) & set(new_top_idx.tolist()))
    print(f" top-{topk} overlap: {overlap}/{topk}")

    return {
        "old_prob": old_prob,
        "new_prob": new_prob,
        "old_logit": old_logit,
        "new_logit": new_logit,
        "corr": corr,
        "old_top_idx": old_top_idx,
        "new_top_idx": new_top_idx,
        "topk_overlap": overlap,
    }



if __name__ == '__main__':
    old_feat = np.load('C:/Users/jplabuser/Downloads/UCF_Test_ten_i3d/UCF_Test_ten_i3d/Abuse028_x264_i3d.npy')
    new_feat = np.load('output/samplevideos/Abuse028_x264.npy')
    summarize_feat(old_feat, "old")
    summarize_feat(new_feat, "new")

    compare_feat_cosine(old_feat, new_feat)

    res = compare_detector_scores(
        old_feat=old_feat,
        new_feat=new_feat,
        model=model,
        device=device,
        topk=5
    )
