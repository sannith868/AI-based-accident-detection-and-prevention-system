import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_objects(image):
    results = model(image)

    # Default values
    measurements = {
        "body_length": 0,
        "height": 0,
        "leg_length": 0,
        "chest_width": 0,
        "rump_angle": 0,
        "breed": "Unknown"
    }

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            width = x2 - x1
            height = y2 - y1

            # Measurements
            body_length = width
            body_height = height
            leg_length = int(height * 0.4)
            chest_width = int(width * 0.6)
            rump_angle = int(np.degrees(np.arctan2(height, width)))

            # Breed logic
            if height > 300:
                breed = "Large Breed (Cow)"
            elif height > 200:
                breed = "Medium Breed"
            else:
                breed = "Small Breed"

            # Store values
            measurements["body_length"] = body_length
            measurements["height"] = body_height
            measurements["leg_length"] = leg_length
            measurements["chest_width"] = chest_width
            measurements["rump_angle"] = rump_angle
            measurements["breed"] = breed

            # Draw text
            cv2.putText(image, f"Breed: {breed}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image, measurements