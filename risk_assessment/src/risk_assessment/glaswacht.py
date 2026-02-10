
from textual.app import App
from textual.widgets import Static, Header
from textual.containers import Container
from risk_assessment.orchestrator import Orchestrator
import threading



LOGO = """
░██████╗░██╗░░░░░░█████╗░░██████╗░██╗░░░░░░░██╗░█████╗░░█████╗░██╗░░██╗████████╗
██╔════╝░██║░░░░░██╔══██╗██╔════╝░██║░░██╗░░██║██╔══██╗██╔══██╗██║░░██║╚══██╔══╝
██║░░██╗░██║░░░░░███████║╚█████╗░░╚██╗████╗██╔╝███████║██║░░╚═╝███████║░░░██║░░░
██║░░╚██╗██║░░░░░██╔══██║░╚═══██╗░░████╔═████║░██╔══██║██║░░██╗██╔══██║░░░██║░░░
╚██████╔╝███████╗██║░░██║██████╔╝░░╚██╔╝░╚██╔╝░██║░░██║╚█████╔╝██║░░██║░░░██║░░░
░╚═════╝░╚══════╝╚═╝░░╚═╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░
"""

INFO = """
Requirements for Glaswacht: Geforce 4090 RTX (with 24GB VRAM) or better.
"""

class ModelPanel(Static):
    def update_state(self, state):
        self.update(f"""
[bold]{state.name}[/bold]
Status: {state.status.value}
Task: {state.current_task}
        """)

class MonitorApp(App):
    CSS = """
    ModelPanel {
        border: solid green;
        margin: 1;
        padding: 1;
    }
    """
    
    def __init__(self, orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.panels = {}
    
    def compose(self):
        yield Header()
        yield Container(id="models")
    
    def on_mount(self):
        models_container = self.query_one("#models")
        
        # Create a panel for each model
        for model_id, state in self.orchestrator.get_state().items():
            panel = ModelPanel()
            panel.update_state(state)
            self.panels[model_id] = panel
            models_container.mount(panel)
        
        # Start models in background
        threading.Thread(target=self.orchestrator.run_all, daemon=True).start()
        
        # Update display every 100ms
        self.set_interval(0.1, self.update_display)
    
    def update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run()