import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from ultralytics import YOLO
import cv2
import numpy as np
import os
import winsound
from datetime import datetime  # ⏳ Timestamp

# 📹 Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise ValueError("Could not open webcam.")

# 📏 Get video resolution
w, h = (
    int(video_capture.get(x))
    for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
    )
)

# 💾 Output video settings (timestamped filename)
output_folder = "./ResultsVideoYOLO"
os.makedirs(output_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
res_path = f"webcam_output_{timestamp}.mp4"

result_video = cv2.VideoWriter(
    os.path.join(output_folder, res_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (w, h)
)

# 🧠 Load YOLO pose model (silent mode)
inferencer = YOLO("yolo11n-pose.pt")
confidence_threshold = 0.5
frames_required = 5

# Tracking variables
alert_frame_count = 0
danger_zone = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Run inference silently
    results = inferencer(frame, verbose=False)
    nb_detected = 0

    if results:
        result = results[0]
        keypoints = result.keypoints

        if keypoints and keypoints.xy is not None:
            person_count = keypoints.xy.shape[0]
            person_boxes = []

            for person_idx in range(person_count):
                person_conf = keypoints.conf[person_idx].mean().item()

                if person_conf >= confidence_threshold:
                    x_min = int(keypoints.xy[person_idx][:, 0].min())
                    y_min = int(keypoints.xy[person_idx][:, 1].min())
                    x_max = int(keypoints.xy[person_idx][:, 0].max())
                    y_max = int(keypoints.xy[person_idx][:, 1].max())
                    area = (x_max - x_min) * (y_max - y_min)
                    person_boxes.append((area, (x_min, y_min, x_max, y_max), person_idx))

            if person_boxes:
                person_boxes.sort(reverse=True)
                _, main_user_box, main_user_idx = person_boxes[0]
                mx1, my1, mx2, my2 = main_user_box

                dz_width = mx2 - mx1
                dz_height = my2 - my1
                danger_zone = (
                    max(0, mx1 - dz_width // 2),
                    max(0, my1),
                    min(w, mx2 + dz_width // 2),
                    min(h, my2 + dz_height // 2)
                )

                for _, _, person_idx in person_boxes[1:]:
                    hips = keypoints.xy[person_idx][[11, 12]]
                    mid_x = int((hips[0][0] + hips[1][0]) / 2)
                    mid_y = int((hips[0][1] + hips[1][1]) / 2)

                    if danger_zone[0] <= mid_x <= danger_zone[2] and danger_zone[1] <= mid_y <= danger_zone[3]:
                        nb_detected += 1

        res = result.plot()
    else:
        res = frame.copy()

    if danger_zone:
        cv2.rectangle(res, (danger_zone[0], danger_zone[1]),
                      (danger_zone[2], danger_zone[3]),
                      (0, 0, 255), 2)
        cv2.putText(res, "Danger Zone", (danger_zone[0], danger_zone[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if nb_detected >= 1:
        alert_frame_count += 1
    else:
        alert_frame_count = 0

    if alert_frame_count >= frames_required:
        winsound.Beep(1000, 300)
        alert_frame_count = 0

    cv2.putText(
        res, f'Intruders in zone: {nb_detected}', (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
    )

    cv2.imshow("YOLO Adaptive Danger Zone", res)
    result_video.write(res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
result_video.release()
cv2.destroyAllWindows()

print(f"✅ Video saved to: {os.path.join(output_folder, res_path)}")
