import cv2
from ultralytics import YOLO
import tensorflow as tf

# Load a larger YOLOv8 model for better accuracy
model = YOLO('yolov8m.pt')  # You can use 'yolov8m.pt', 'yolov8l.pt', etc., for more accuracy

# Initialize the video capture
cap = cv2.VideoCapture(0)

with tf.device('/device:GPU:0'):  # Ensuring we use the GPU if available
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Extract detection results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            # Draw bounding boxes and labels on the frame
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_id = int(classes[i])
                label = model.names[class_id]

                if score > 0.5:  # Adjusted threshold for more confident detections
                    # Extract box coordinates
                    start_x, start_y, end_x, end_y = map(int, box[:4])

                    # Draw the bounding box
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

                    # Draw the label and score
                    label_text = f"{label}: {score:.2f}"
                    cv2.putText(frame, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv8 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
