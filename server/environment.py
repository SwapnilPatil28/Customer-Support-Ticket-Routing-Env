import uuid
from typing import List, Dict
from openenv.core.env_server import Environment
from models import SREAction, SREObservation, SREState

class SREEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.logs = {
            "auth-service": "ERROR: Connection to DB timed out.",
            "database": "WARN: Storage 99% full.",
            "frontend": "500 Internal Server Error calling backend-service",
            "backend-service": "Failed to authenticate request: timeout from auth-service"
        }
        self.metrics = {
            "dash-db": "CPU: 15%, Memory: 99.9%, Disk: 99% full.",
            "dash-auth": "CPU: 5%",
            "dash-front": "CPU: 10%",
            "dash-back": "CPU: 20%"
        }
        self.tasks = {
            "easy": [{"id": "E1", "text": "Users can't login, check auth-service logs.", "root_cause": "database"}],
            "medium": [{"id": "M1", "text": "Backend service failing.", "root_cause": "auth-service"}],
            "hard": [{"id": "H1", "text": "Frontend reporting 500s. Trace it back to the root cause.", "root_cause": "database"}],
        }

    def reset(self, task_name: str = "easy") -> SREObservation:
        self.current_task = self.tasks.get(task_name, self.tasks["easy"])
        self._state = SREState(episode_id=str(uuid.uuid4()), task_id=task_name, current_ticket_index=0)
        t = self.current_task[0]
        return SREObservation(done=False, reward=0.0, ticket_id=t["id"], content=t["text"], terminal_output="Environment initialized.")

    def step(self, action: SREAction) -> SREObservation:
        self._state.step_count += 1
        ticket = self.current_task[self._state.current_ticket_index]

        reward = 0.0
        terminal_output = ""

        if action.action_type == "query_logs":
            reward = -0.1
            terminal_output = self.logs.get(action.service_name, f"No logs found for {action.service_name}")
        elif action.action_type == "check_metrics":
            reward = -0.1
            terminal_output = self.metrics.get(action.dashboard_id, f"Dashboard {action.dashboard_id} not found")
        elif action.action_type == "resolve_ticket":
            correct = action.root_cause and action.root_cause.strip().lower() == ticket["root_cause"].lower()
            reward = 1.0 if correct else -1.0
            
            self._state.current_ticket_index += 1
            if self._state.current_ticket_index < len(self.current_task):
                t = self.current_task[self._state.current_ticket_index]
                return SREObservation(done=False, reward=reward, ticket_id=t["id"], content=t["text"], terminal_output=f"Previous ticket resolved. Result: {'Correct' if correct else 'Incorrect'}. Next ticket.")
            
            return SREObservation(done=True, reward=reward, ticket_id="EOF", content="Done.", terminal_output=f"Final ticket resolved. Result: {'Correct' if correct else 'Incorrect'}.")

        return SREObservation(done=False, reward=reward, ticket_id=ticket["id"], content=ticket["text"], terminal_output=terminal_output)

    @property
    def state(self) -> SREState:
        return self._state
