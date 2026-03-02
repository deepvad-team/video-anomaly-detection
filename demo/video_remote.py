# A에서 실행

import cv2

B_IP = "100.102.31.119"  # B의 tailscale IP
cap = cv2.VideoCapture(f"http://{B_IP}:5000/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 여기서 YOLO / VAD / I3D inference 넣으면 됨

    cv2.imshow("stream", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()