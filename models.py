from typing import Optional, Literal
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class SREAction(Action):
    action_type: Literal["query_logs", "check_metrics", "resolve_ticket"] = Field(..., description="Action to take")
    service_name: Optional[str] = Field(None, description="Required for query_logs")
    dashboard_id: Optional[str] = Field(None, description="Required for check_metrics")
    root_cause: Optional[str] = Field(None, description="Required for resolve_ticket")

class SREObservation(Observation):
    ticket_id: str
    content: str
    terminal_output: str = ""

class SREState(State):
    task_id: str = "easy"
    current_ticket_index: int = 0
