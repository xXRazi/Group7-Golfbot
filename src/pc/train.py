from ultralytics import YOLO

def train_robot_vision():
    # 1. Load a pre-trained base model
    # We use 'yolov8n.pt' (Nano). It is the smallest and fastest model, 
    # making it perfect for running in real-time on robotics hardware like a Raspberry Pi.
    # The script will automatically download this file from the internet the first time you run it.
    model = YOLO('yolov8n.pt')

    print("Starting training process...")

    # 2. Train the model
    # IMPORTANT: Update the 'data' path to point to your specific data.yaml file!
    results = model.train(
        data='/home/sander/Downloads/golfbot.v8i.yolov8/data.yaml', 
        epochs=50,             # Number of times the AI will look through your entire dataset
        patience=15,           # Stop training if the model doesn't improve for 15 epochs
        imgsz=640,             # Standard image resolution YOLO uses to learn
        batch=16,              # How many images it loads into memory at once
        device='cpu',          # What hardware to use for training
        name='arena_model_v1'  # The name of the folder where it will save your results
    )

    print("Training complete!")

if __name__ == '__main__':
    train_robot_vision()