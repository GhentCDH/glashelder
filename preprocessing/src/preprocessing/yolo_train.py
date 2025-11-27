from ultralytics import YOLO

# Train model

def train_yolo_model(model, data_yaml, epochs, project_folder, project_name):
    model = YOLO(model) 
    results = model.train(data=data_yaml, epochs=epochs, imgsz=1028, verbose=True, project=project_folder, name=project_name, exist_ok=True)