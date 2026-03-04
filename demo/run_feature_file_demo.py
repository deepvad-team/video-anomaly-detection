# demo/run_feature_file_demo.py
import torch
import numpy as np

from feature_bridge.feature_loader import load_feature_npy, feature_to_tensor, summarize_feature
from feature_bridge.detector_runner import run_detector_on_feature, summarize_scores

from model import Model_V2   # 네 프로젝트 경로에 맞게 수정


def load_model(ckpt_path: str, device: torch.device, feature_size: int = 2048):
    model = Model_V2(feature_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_path = "C:/Users/jplabuser/Downloads/UCF_Test_ten_i3d/UCF_Test_ten_i3d/Arrest024_x264_i3d.npy"  # extractor가 만든 파일 경로로 수정
    ckpt_path = "../../minjeong/unsupervised_ckpt/UCF_final_20260301_031008_pgn3aode.pkl"

    feat = load_feature_npy(feature_path)
    summarize_feature(feat, "input_feature")

    x = feature_to_tensor(feat, device)
    model = load_model(ckpt_path, device, feature_size=2048)

    prob, logit = run_detector_on_feature(x, model)
    summarize_scores(prob, logit, "detector_output")

    print("prob[:10] =", prob[:10])
    print("logit[:10] =", logit[:10])