import logging

logger = logging.getLogger(__name__)

# Train model

def train_yolo_model(model, data_yaml, epochs, project_folder, project_name):
    from ultralytics import YOLO

    logger.info(f"Starting YOLO training with model: {model}, data: {data_yaml}, epochs: {epochs}")
    model = YOLO(model) 
    results = model.train(data=data_yaml, epochs=epochs, imgsz=1028, verbose=True, project=project_folder, name=project_name, exist_ok=True)
    logger.info("YOLO training completed")