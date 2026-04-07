import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from models import Action
from server.environment import ReturnTriageEnv

app = FastAPI(title="OpenEnv: E-Commerce Returns Triage")
current_env = None

class ResetRequest(BaseModel):
    task_level: Optional[str] = "hard"
@app.get("/")
def health():
    return {"status": "ok"}
    
@app.post("/reset")
def reset_environment(request: Optional[ResetRequest] = Body(None)):
    global current_env
    requested_task = request.task_level if request and request.task_level else "hard"
    
    # Path safety for 4 tasks
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_file = os.path.join(base_dir, "tasks", f"{requested_task}_01.json")
    
    if not os.path.exists(task_file):
        requested_task = "hard"
    
    try:
        current_env = ReturnTriageEnv(task_level=requested_task)
        observation = current_env.reset()
        return {"status": "success", "observation": observation.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    global current_env
    if current_env is None:
        current_env = ReturnTriageEnv(task_level="hard")
        current_env.reset()
        
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