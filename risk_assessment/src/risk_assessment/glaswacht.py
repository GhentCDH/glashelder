
from textual.app import App
from textual.widgets import Static, Header
from textual.containers import Container, Vertical, VerticalScroll
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
        rating_display = f"[cyan]{state.latest_rating}[/cyan]" if state.latest_rating else ""
        image_display = f"[dim]{state.latest_image}[/dim]" if state.latest_image else ""
        
        self.update(f"""
[bold]{state.name}[/bold]
Status: {state.status.value}
Task: {state.current_task}
{image_display}
{rating_display}
        """)

class MajorityVotePanel(Static):
    def update_votes(self, majority_votes):
        if not majority_votes:
            self.update("[bold]Majority Votes[/bold]\n[dim]No results yet...[/dim]")
            return
        
        # Count votes by category
        vote_counts = {"Unsafe": 0, "Potentially Problematic": 0, "Safe": 0, "Other": 0}
        for vote in majority_votes.values():
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts["Other"] += 1
        
        # Build summary
        summary_parts = []
        if vote_counts["Unsafe"] > 0:
            summary_parts.append(f"[red]{vote_counts['Unsafe']} Unsafe[/red]")
        if vote_counts["Potentially Problematic"] > 0:
            summary_parts.append(f"[yellow]{vote_counts['Potentially Problematic']} Potentially Problematic[/yellow]")
        if vote_counts["Safe"] > 0:
            summary_parts.append(f"[green]{vote_counts['Safe']} Safe[/green]")
        if vote_counts["Other"] > 0:
            summary_parts.append(f"[white]{vote_counts['Other']} Other[/white]")
        
        summary = " | ".join(summary_parts) if summary_parts else "No votes"
        
        vote_lines = [f"[bold]Majority Votes[/bold] ({len(majority_votes)} images)\n{summary}\n"]
        
        for image_name, vote in majority_votes.items():
            # Color code based on vote
            if vote == "Unsafe":
                color = "red"
            elif vote == "Potentially Problematic":
                color = "yellow"
            elif vote == "Safe":
                color = "green"
            else:
                color = "white"
            
            vote_lines.append(f"[dim]{image_name}[/dim]: [{color}]{vote}[/{color}]")
        
        self.update("\n".join(vote_lines))

class MonitorApp(App):
    CSS = """
    ModelPanel {
        border: solid green;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    MajorityVotePanel {
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    #models {
        height: auto;
    }
    
    #results-scroll {
        height: 1fr;
        max-height: 50%;
        border: solid blue;
    }
    """
    
    def __init__(self, orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.panels = {}
        self.majority_panel = None
    
    def compose(self):
        yield Header()
        yield Vertical(
            Container(id="models"),
            VerticalScroll(id="results-scroll")
        )
    
    def on_mount(self):
        models_container = self.query_one("#models")
        results_container = self.query_one("#results-scroll")
        
        # Create a panel for each model
        for model_id, state in self.orchestrator.get_state().items():
            panel = ModelPanel()
            panel.update_state(state)
            self.panels[model_id] = panel
            models_container.mount(panel)
        
        # Create majority vote panel
        self.majority_panel = MajorityVotePanel()
        self.majority_panel.update_votes({})
        results_container.mount(self.majority_panel)
        
        # Start models in background
        threading.Thread(target=self.orchestrator.run_all, args=("//mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop",), daemon=True).start()
        
        # Update display every 100ms
        self.set_interval(0.1, self.update_display)
    
    def update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)
        
        # Update majority votes
        majority_votes = self.orchestrator.get_majority_votes()
        if self.majority_panel:
            self.majority_panel.update_votes(majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run()