# demo/webcam_demo.py
import sys
from collections import deque

import cv2
import time
import torch
import numpy as np

# === extractor project 경로 추가 ===
sys.path.append(r"C:\\Users\\jplabuser\\I3D_Feature_Extraction_resnet")

from bridge_api import load_i3d_model

from demo.extractor_bridge import extract_clip_feature
from adapter import ResidualAdapter2048
from demo.demo_utils import run_detector_with_adapter, EMASmoother, HysteresisAlert
from demo.demo_tea import demo_initial_tea_adapt
from model import Model_V2

def load_detector(ckpt_path, device, feature_size=2048):
    model = Model_V2(feature_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) extractor model
    i3d_model = load_i3d_model(
        r"C:\\Users\\jplabuser\\I3D_Feature_Extraction_resnet\\pretrained\\i3d_r50_kinetics.pth"
    )
    print("I3D model loaded.")


    # 2) detector
    detector = load_detector(
        r"C:\\Users\\jplabuser\\minjeong\\unsupervised_ckpt\\UCF_final_20260301_031008_pgn3aode.pkl",
        device=device,
        feature_size=2048
    )
    print("I3D model loaded.")

    # 3) adapter
    adapter = ResidualAdapter2048(d=2048, use_ln=True).to(device)
    adapter.eval()
    print("Detector loaded.")

    # 4) webcam
    B_IP = "100.102.31.119"  # B의 tailscale IP
    cap = cv2.VideoCapture(f"http://{B_IP}:5000/video")
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    clip_len = 16
    clip_stride = 16      # 추론은 16프레임마다 한 번
    sample_every = 2      # 웹캠 프레임은 2프레임에 1번만 버퍼에 넣기
    frame_buffer = deque(maxlen=clip_len)

    frame_counter = 0
    infer_counter = 0

    # calibration / TEA
    calibrating = True
    calibration_clips = 8
    calib_features = []
    calib_logits_before = []

    # after calibration
    mu = 0.0
    sigma = 1.0

    # display state
    raw_logit_last = None
    prob_last = None
    z_last = None
    smooth_z = EMASmoother(alpha=0.2)
    alert_state = HysteresisAlert(enter_thr=2.0, exit_thr=1.2)
    
    status_text = "BUFFERING"
    raw_text = "raw logit: -"
    prob_text = "prob: -"
    z_text = "z-score: -"
    #smooth_text = "smooth logit: -"
    info_text = "waiting for enough frames..."

    print("Starting webcam demo... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # 일부 프레임만 버퍼에 저장 (frame downsample)
        if frame_counter % sample_every == 0:
            frame_buffer.append(frame)

        vis = frame.copy()

        # 버퍼가 아직 안 찼으면 상태만 유지
        if len(frame_buffer) < clip_len:
            status_text = "BUFFERING"
            info_text = f"collecting clip frames: {len(frame_buffer)}/{clip_len}"
        else:
            # clip_stride마다 한 번만 추론
            if frame_counter % clip_stride == 0:
                infer_counter += 1
                frames = list(frame_buffer)
                status_text = "PROCESSING"
                #info_text = f"inference #{infer_counter}"
            try:
                # === extractor -> (1,10,2048) ===
                feat = extract_clip_feature(frames, i3d_model, sample_mode='oversample')

                # === current detector output ===
                prob, logit = run_detector_with_adapter(feat, adapter, detector, device)

                raw_logit = float(logit[-1])
                raw_prob = float(prob[-1])

                raw_logit_last = raw_logit
                prob_last = raw_prob

                
                if calibrating:
                    calib_features.append(feat)          # each feat: (1,10,2048)
                    calib_logits_before.append(raw_logit)

                    status_text = f"CALIBRATING {len(calib_features)}/{calibration_clips}"
                    raw_text = f"raw logit: {raw_logit_last:.4f}"
                    prob_text = f"prob: {prob_last:.6f}"
                    z_text = "z-score: calibrating"
                    info_text = "collecting normal buffer"

                    # enough clips -> initial TEA + calibration stats recompute
                    if len(calib_features) >= calibration_clips:
                        calib_feat_np = np.concatenate(calib_features, axis=0)   # (N,10,2048)
                        calib_feat_tensor = torch.from_numpy(calib_feat_np).float().to(device)

                        # initial TEA on normal buffer only
                        adapter = demo_initial_tea_adapt(
                            normal_feat_tensor=calib_feat_tensor,
                            adapter=adapter,
                            model=detector,
                            tea_steps=2,
                            sgld_steps=10,
                            sgld_lr=0.05,
                            sgld_noise=0.01,
                            tea_lr=1e-3,
                        )

                        # recompute logits after adaptation to define new normal baseline
                        _, calib_logit_after = run_detector_with_adapter(
                            calib_feat_np, adapter, detector, device
                        )
                        mu = float(np.mean(calib_logit_after))
                        sigma = float(np.std(calib_logit_after) + 1e-6)

                        calibrating = False
                        smooth_z = EMASmoother(alpha=0.2)
                        alert_state = HysteresisAlert(enter_thr=2.0, exit_thr=1.2)

                        print(f"[Calibration done] mu={mu:.4f}, sigma={sigma:.4f}")

                else:
                    z = (raw_logit - mu) / sigma
                    z_last = z
                    z_smooth = smooth_z.update(z)
                    alert = alert_state.update(z_smooth)

                    status_text = "ALERT" if alert else "NORMAL"
                    raw_text = f"raw logit: {raw_logit_last:.4f}"
                    prob_text = f"prob: {prob_last:.6f}"
                    z_text = f"z-score: {z_smooth:.4f}"
                    info_text = f"infer #{infer_counter} | mu={mu:.3f}, sigma={sigma:.3f}"

                    color = (0, 0, 255) if alert else (0, 255, 0)
                    cv2.rectangle(vis, (10, 10), (360, 140), color, 3)

            except Exception as e:
                status_text = "ERROR"
                raw_text = str(e)[:60]
                info_text = "inference failed"

    # === 화면 표시 ===
        cv2.putText(vis, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis, raw_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, prob_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, z_text, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"buffer: {len(frame_buffer)}/{clip_len}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, info_text, (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("webcam_demo", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()