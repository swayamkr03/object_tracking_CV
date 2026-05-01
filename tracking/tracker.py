import math

class Tracker:
    def __init__(self):
        # Store object ID -> (x, y)
        self.objects = {}
        self.id_count = 0

    def update(self, detections):
        updated_objects = {}

        for det in detections:
            x1, y1, x2, y2 = det

            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            same_object_detected = False

            for obj_id, (px, py) in self.objects.items():
                distance = math.hypot(cx - px, cy - py)

                # 🔥 Distance threshold (tune if needed)
                if distance < 50:
                    updated_objects[obj_id] = (cx, cy)
                    same_object_detected = True
                    break

            # New object
            if not same_object_detected:
                updated_objects[self.id_count] = (cx, cy)
                self.id_count += 1

        # Update tracked objects
        self.objects = updated_objects

        return self.objects