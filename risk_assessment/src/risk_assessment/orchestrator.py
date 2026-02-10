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
    latest_rating: str = ""
    latest_image: str = ""


class Orchestrator:
    def __init__(self):
        self.models = {
            "model1": ModelState("LavaGuard", ModelStatus.CHILLING),
            "model2": ModelState("Qwen3", ModelStatus.CHILLING),
            "model3": ModelState("Pixtral", ModelStatus.CHILLING),
        }
        self.assessments_lavaguard = []
        self.assessments_qwen = []
        self.assessments_pixtral = []
        self.majority_votes = {}
        self._lock = __import__('threading').Lock()

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
    
    def update_majority_votes_live(self):
        """Recalculate majority votes based on current assessments."""
        from risk_assessment.models.utils import calculate_majority_vote
        
        with self._lock:
            # Create dictionaries keyed by image name
            lava_dict = {a.get("image"): a.get("rating", "Unknown") for a in self.assessments_lavaguard}
            qwen_dict = {a.get("image"): a.get("rating", "Unknown") for a in self.assessments_qwen}
            pixtral_dict = {a.get("image"): a.get("rating", "Unknown") for a in self.assessments_pixtral}
            
            # Get all image names
            all_images = set(lava_dict.keys()) | set(qwen_dict.keys()) | set(pixtral_dict.keys())
            
            # Calculate majority for images that have all three assessments
            for image_name in all_images:
                if image_name in lava_dict and image_name in qwen_dict and image_name in pixtral_dict:
                    majority = calculate_majority_vote(
                        lava_dict[image_name],
                        qwen_dict[image_name],
                        pixtral_dict[image_name]
                    )
                    self.majority_votes[image_name] = majority
    
    def run_all(self):
        from risk_assessment.models.model_LLavaguard import run_lavaguard
        from risk_assessment.models.model_Qwen3 import run_qwen
        from risk_assessment.models.model_Pixtral import run_pixtral
        from risk_assessment.models.utils import write_combined_csv

        image_path = "/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/sn01004006_tiff.jpg"
        
        run_lavaguard(self, image_path)
        run_qwen(self, image_path)
        run_pixtral(self, image_path)
        
        # Write combined results to CSV
        write_combined_csv(self.assessments_lavaguard, self.assessments_qwen, self.assessments_pixtral)
    
    def get_majority_votes(self):
        """Get the dictionary of majority votes by image name."""
        with self._lock:
            return self.majority_votes.copy()
    



if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_all()