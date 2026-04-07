import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional

# Absolute imports from the root directory
from models import Action
from server.environment import ReturnTriageEnv

app = FastAPI(title="OpenEnv: E-Commerce Returns Triage")
current_env = None

class ResetRequest(BaseModel):
    task_level: Optional[str] = "hard"

# Health check for Hugging Face UI
@app.get("/")
def health():
    return {"status": "ok", "message": "OpenEnv Returns Triage Server is Running"}

@app.post("/reset")
def reset_environment(request: Optional[ResetRequest] = Body(None)):
    global current_env
    requested_task = request.task_level if request and request.task_level else "hard"
    
    # Find tasks relative to the app's root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_file = os.path.join(base_dir, "tasks", f"{requested_task}_01.json")
    
    if not os.path.exists(task_file):
        requested_task = "hard" # Safety fallback
    
    try:
        current_env = ReturnTriageEnv(task_level=requested_task)
        observation = current_env.reset()
        return {"status": "success", "observation": observation.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    global current_env
    
    # Auto-initialize if /reset was skipped (Resilience Fix)
    if current_env is None:
        try:
            current_env = ReturnTriageEnv(task_level="hard")
            current_env.reset()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-init failed: {str(e)}")
            
    try:
        obs, reward, done, info = current_env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": float(reward),
            "done": bool(done),
            "info": jsonable_encoder(info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()