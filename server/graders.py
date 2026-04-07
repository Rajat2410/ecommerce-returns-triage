def calculate_reward(state, action) -> tuple[float, bool]:
    # CLEVER TASK LOGIC
    if state.task_level == "clever":
        # The agent MUST use NO_RETURN_REFUND to get the full score
        if action.action_type == "NO_RETURN_REFUND":
            # Check if they also got the refund amount right (8.50)
            if abs(action.refund_amount - ground_truth_math) < 0.01:
                return 1.0  # Perfect Score
            return 0.5  # Right action, wrong math
        
        elif action.action_type == "ISSUE_REFUND":
            return 0.2  # They gave money back, but wasted $12 on shipping!
            
        return 0.0 # Denied or wrong action
    
    # Penalty: Issuing refund before verifying damage
    if action.action_type == "ISSUE_REFUND":
        item_condition = state.observation.items[0].condition
        history_str = str(state.observation.conversation_history)
        if item_condition == "damaged_claimed" and "REQUEST_PHOTO" not in history_str:
            return -0.5, False 
            
    # Business Logic Trade-off
    if action.action_type == "NO_RETURN_REFUND":
        item_price = state.observation.items[0].price
        if item_price < 10.00:
            return 0.5, True 
        else:
            return -1.0, False 

    # Base reward for a structurally valid intermediate step
    if action.action_type == "ASK_QUESTION":
        return 0.1, True
        
    return 0.0, True