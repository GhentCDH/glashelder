from enum import Enum

from dataclasses import dataclass

class ModelStatus(Enum):
    CHILLING = "Chilling 😎"
    LOADING = "Loading ⚙️"
    RUNNING = "Running 👀"
    DONE = "Done ✅"
    ERROR = "Error ❌"

@dataclass
class ModelState:
    name: str
    status: ModelStatus
    current_task: str = ""


class Orchestrator:
    def __init__(self):
        self.models = {
            "model1": ModelState("model1", ModelStatus.CHILLING),
            "model2": ModelState("model2", ModelStatus.CHILLING),
            "model3": ModelState("model3", ModelStatus.CHILLING),
        }

    def update_status(self, model_name: str, new_status: ModelStatus):
        if model_name in self.models:
            self.models[model_name] = new_status
            print(f"{model_name} status updated to: {new_status.value}")
        else:
            print(f"Model {model_name} not found.")
    
    def get_state(self):
        return self.models.copy()
    
    def update_model_state(self, model_id: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self.models[model_id], key, value)
    
    def run_all(self):
        from risk_assessment.models.model_LLavaguard import run_lavaguard
        from risk_assessment.models.model_Qwen3 import run_qwen
        from risk_assessment.models.model_Pixtral import run_pixtral
        from risk_assessment.models.utils import write_combined_csv

        image_path = "/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/sn01004006_tiff.jpg"
        
        assessments_lavaguard = run_lavaguard(self, image_path)
        assessments_qwen = run_qwen(self, image_path)
        assessments_pixtral = run_pixtral(self, image_path)
        
        # Write combined results to CSV
        write_combined_csv(assessments_lavaguard, assessments_qwen, assessments_pixtral)
    



if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_all()