from ultralytics import YOLO

# Train model

def train_yolo_model(data_yaml):
    model = YOLO("preprocessing_utils/data/yolo11m.pt/yolo11m.pt") 
    results = model.train(data=data_yaml, epochs=30, imgsz=1028, verbose=True, project="preprocessing_utils/data/models", name="preproces_model",exist_ok=True)

train_yolo_model("preprocessing_utils/data/training_data/data.yaml")