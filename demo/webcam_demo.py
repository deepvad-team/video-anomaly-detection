# demo/webcam_demo.py
import sys
import cv2
import time
from collections import deque
import torch

sys.path.append(r"C:\\Users\\jplabuser\\I3D_Feature_Extraction_resnet")

from bridge_api import load_i3d_model
from demo.extractor_bridge import extract_clip_feature
from adapter import ResidualAdapter2048
from demo.demo_utils import run_detector_with_adapter, EMASmoother, HysteresisAlert
#from demo.demo_tea import demo_initial_tea_adapt
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

    raw_logit_last = None
    prob_last = None
    smooth_logit = EMASmoother(alpha=0.2)

    print("Starting webcam demo... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        # 일부 프레임만 버퍼에 저장
        if frame_counter % sample_every == 0:
            frame_buffer.append(frame)

        vis = frame.copy()
        status_text = "BUFFERING"
        raw_text = "raw logit: -"
        prob_text = "prob: -"
        smooth_text = "smooth logit: -"
        info_text = "waiting for enough frames..."

        # 버퍼가 아직 안 찼으면 상태만 유지
        if len(frame_buffer) < clip_len:
            status_text = "BUFFERING"
            info_text = f"collecting clip frames: {len(frame_buffer)}/{clip_len}"
        else:
            # clip_stride마다 한 번만 추론
            if frame_counter % clip_stride == 0:
                infer_counter += 1
                status_text = "PROCESSING"
                info_text = f"inference #{infer_counter}"
            try:
                frames = list(frame_buffer)
                # === extractor -> (1,10,2048) ===
                feat = extract_clip_feature(frames, i3d_model, sample_mode='oversample')

                # === detector ===
                prob, logit = run_detector_with_adapter(feat, adapter, detector, device)

                raw_logit_last = float(logit[-1])
                prob_last = float(prob[-1])
                smooth_val = smooth_logit.update(raw_logit_last)

                status_text = "RUNNING"
                raw_text = f"raw logit: {raw_logit_last:.4f}"
                prob_text = f"prob: {prob_last:.6f}"
                smooth_text = f"smooth logit: {smooth_val:.4f}"

            except Exception as e:
                status_text = "ERROR"
                raw_text = str(e)[:60]

        # === 화면 표시 ===
        cv2.putText(vis, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis, raw_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, prob_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, smooth_text, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"buffer: {len(frame_buffer)}/{clip_len}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("webcam_demo", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()