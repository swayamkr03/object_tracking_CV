import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import random

# 🔥 Unique color per ID
def get_color(id):
    random.seed(id)
    return (
        random.randint(50,255),
        random.randint(50,255),
        random.randint(50,255)
    )

def process_video(video_path):

    print("Processing:", video_path)

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return None

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 20

    os.makedirs("output", exist_ok=True)
    output_path = "output/output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # 🚗 Vehicle filter
            if conf > 0.5 and cls in [2, 3, 5, 7]:
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        active_tracks = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            active_tracks += 1

            track_id = track.track_id
            cls = track.det_class

            l, t, r, b = map(int, track.to_ltrb())
            color = get_color(track_id)

            total_ids.add(track_id)

            label = model.names[cls].upper()
            text = f"{label}  ID:{track_id}"

            # 🔥 Bounding box
            cv2.rectangle(frame, (l, t), (r, b), color, 2)

            # 🔥 Text background
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )

            cv2.rectangle(frame,
                          (l, t - th - 10),
                          (l + tw + 10, t),
                          color,
                          -1)

            # 🔥 Text
            cv2.putText(frame, text,
                        (l + 5, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2)

        # 🔥 Transparent HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (260,110), (0,0,0), -1)

        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, f"TOTAL: {len(total_ids)}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2)

        cv2.putText(frame, f"ACTIVE: {active_tracks}",
                    (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,255),
                    2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved at:", os.path.abspath(output_path))

    return output_path


if __name__ == "__main__":
    process_video("data/input.mp4")