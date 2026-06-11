import cv2
from ultralytics import YOLO

def test_live_vision():
    # 1. Load your newly trained "brain"
    model_path = "/Users/jacobsoegaard/PycharmProjects/Group7-Golfbot/runs/detect/arena_model_v1/weights/best.pt"
    model = YOLO(model_path)

    # 2. Show the model classes that are available for detection.
    print("Found these classes in your model:", model.names)

    # 3. Open the Webcam 
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera failed to open. Running YOLO detection on 0.jpg instead.")
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "0.jpg")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not read 0.png")
            return
        
        # Run detection on the image
        results = model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )
        
        # Print detections
        detections = results[0]
        if len(detections.boxes) > 0:
            print("Detected objects:")
            for box in detections.boxes:
                class_id = int(box.cls[0])
                class_name = detections.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                print(f"  - {class_name}: conf={confidence:.2f}, bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
        else:
            print("No objects detected")
        
        # Draw and display
        annotated_frame = results[0].plot()
        cv2.imshow("Golfbot Vision Test", annotated_frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("Starting video feed... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        # 4. Run the AI on the current frame and show all detected classes.
        results = model.predict(
            source=frame,
            conf=0.45,
            verbose=False
        )
        # 4b. Print all detections with their details
        detections = results[0]
        detected_items = []
        for i, box in enumerate(detections.boxes):
            class_id = int(box.cls[0])
            class_name = detections.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            detected_items.append({
                'class': class_name,
                'confidence': f"{confidence:.2f}",
                'bbox': f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            })
        
        if detected_items:
            print("Detected objects:")
            for item in detected_items:
                print(f"  - {item['class']}: conf={item['confidence']}, bbox={item['bbox']}")
        else:
            print("No objects detected")
        # 5. Draw the bounding boxes on the image
        annotated_frame = results[0].plot()

        # 6. Show the image on your screen
        cv2.imshow("Golfbot Vision Test", annotated_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_live_vision()