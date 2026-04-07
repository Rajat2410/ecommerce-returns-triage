from pydantic import BaseModel
from typing import List, Optional, Literal

class OrderItem(BaseModel):
    item_id: str
    name: str
    price: float
    category: str
    condition: str

class Observation(BaseModel):
    ticket_id: str
    customer_message: str
    order_date: str
    current_date: str
    items: List[OrderItem]
    policy_snippet: str
    conversation_history: List[dict]

ActionType = Literal[
    "APPROVE_ELIGIBLE", 
    "DENY_INELIGIBLE", 
    "ASK_QUESTION", 
    "ISSUE_REFUND", 
    "NO_RETURN_REFUND"
]

class Action(BaseModel):
    action_type: ActionType
    question_type: Optional[Literal["REQUEST_PHOTO", "CLARIFY_REASON"]] = None
    refund_amount: Optional[float] = None
    reason_code: Optional[str] = None

class EnvState(BaseModel):
    current_step: int
    max_steps: int
    observation: Observation
    is_done: bool
    ground_truth_math: float
    hidden_customer_persona: dict = {}