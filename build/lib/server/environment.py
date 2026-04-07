import json
import os
from models import Observation, Action, EnvState
from server.graders import calculate_reward

class ReturnTriageEnv:
    def __init__(self, task_level: str):
        self.task_level = task_level 
        self.state = None

    def reset(self) -> Observation:
        self.state = self._load_initial_state()
        return self.state.observation

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.state.is_done:
            raise ValueError("Episode already done.")
            
        self.state.current_step += 1
        reward = 0.0
        done = False
        info = {}

        step_reward, valid_action = calculate_reward(self.state, action)
        reward += step_reward

        if action.action_type == "ASK_QUESTION":
            simulated_reply = self._simulate_customer_reply(action.question_type)
            self.state.observation.conversation_history.append({"agent": action.model_dump(), "user": simulated_reply})
            self.state.observation.customer_message = simulated_reply
            
        elif action.action_type in ["APPROVE_ELIGIBLE", "DENY_INELIGIBLE", "ISSUE_REFUND", "NO_RETURN_REFUND"]:
            done = True
            if action.action_type == "ISSUE_REFUND":
                if action.refund_amount == self.state.ground_truth_math:
                    reward += 1.0  
                    info["status"] = "SUCCESS_EXACT_MATH"
                else:
                    reward -= 1.0  
                    info["status"] = "FAILED_MATH"
            else:
                info["status"] = f"TERMINATED_WITH_{action.action_type}"

        if self.state.current_step >= self.state.max_steps and not done:
            done = True
            info["status"] = "TIMEOUT"

        self.state.is_done = done
        return self.state.observation, reward, done, info
        
    def _load_initial_state(self) -> EnvState:
        # Resolves path correctly from within the server directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "tasks", f"{self.task_level}_01.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        return EnvState(
            current_step=0,
            max_steps=data["max_steps"],
            observation=Observation(**data["initial_observation"]),
            is_done=False,
            ground_truth_math=data["ground_truth_math"],
            hidden_customer_persona=data.get("hidden_customer_persona", {})
        )
        
    def _simulate_customer_reply(self, question_type: str) -> str:
        if question_type == "REQUEST_PHOTO":
            return self.state.hidden_customer_persona.get("on_photo_request", "I don't have a camera right now.")
        return "I'm not sure what you mean."