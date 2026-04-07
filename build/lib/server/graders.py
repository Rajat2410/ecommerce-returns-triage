def calculate_reward(state, action) -> tuple[float, bool]:
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