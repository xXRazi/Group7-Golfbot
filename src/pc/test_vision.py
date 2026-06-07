import cv2
from ultralytics import YOLO

def test_live_vision():
    # 1. Load your newly trained "brain"
    model_path = "/home/sander/workspace/CDIO_live/Group7-Golfbot/runs/detect/arena_model_v18/weights/best.pt"
    model = YOLO(model_path)

    # 2. Dynamically find the Class IDs for the balls
    target_classes = []
    
    print("Found these classes in your model:", model.names)
    
    for class_id, class_name in model.names.items():
        # Clean the string to handle spaces, underscores, or dashes
        clean_name = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        
        if clean_name in ["whiteball", "orangeball"]:
            target_classes.append(class_id)

    # 3. Open the Webcam 
    cap = cv2.VideoCapture(1)
    print("Starting video feed... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        # 4. Run the AI on the current frame (The Fix)
        # If target_classes has items, filter by them. If empty, pass None to show everything.
        filter_ids = target_classes if target_classes else None
        
        results = model.predict(
            source=frame, 
            conf=0.5, 
            classes=filter_ids, 
            verbose=False
        )

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