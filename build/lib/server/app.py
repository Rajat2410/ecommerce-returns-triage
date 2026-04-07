from fastapi import FastAPI, HTTPException
from models import Action
from server.environment import ReturnTriageEnv
import uvicorn

app = FastAPI(title="OpenEnv: E-Commerce Returns")
current_env = None

@app.get("/")
def health_check():
    return {"status": "HTTP 200 OK - Space is running!"}

@app.post("/reset")
def reset_environment():
    global current_env
    try:
        current_env = ReturnTriageEnv(task_level="hard")
        return {"status": "success", "observation": current_env.reset().model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        obs, reward, done, info = current_env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ADDITIONS REQUIRED BY VALIDATOR ---
def main():
    """Entry point for the openenv-server command."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()