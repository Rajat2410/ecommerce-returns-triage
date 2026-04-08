import os
from openai import OpenAI
from server.environment import ReturnTriageEnv
from models import Action

# Env variables for the validator
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_baseline(task_name):
    BENCHMARK_NAME = "ecommerce_returns_triage"
    env = ReturnTriageEnv(task_level=task_name)
    obs = env.reset()
    
    done = False
    step_count = 0
    MAX_STEPS = 10
    rewards_log = []
    
    # [START] tag must match exactly
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}")
    
    try:
        while not done and step_count < MAX_STEPS:
            step_count += 1
            
            response = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a strict e-commerce returns agent. Follow policy exactly, ask for missing information when needed, and compute refund amounts precisely."},
                    {"role": "user", "content": f"State: {obs.model_dump_json()}"}
                ],
                response_format=Action,
                temperature=0.0
            )
            
            action_parsed = response.choices[0].message.parsed
            obs, reward, done, info = env.step(action_parsed)
            
            formatted_reward = f"{float(reward):.2f}"
            rewards_log.append(formatted_reward)
            
            # [STEP] tag must be single-line
            print(f"[STEP] step={step_count} action={action_parsed.action_type} reward={formatted_reward} done={str(done).lower()} error=null")

        raw_total = sum(float(r) for r in rewards_log)
        
        # Clamp between 0.01 and 0.99 to strictly satisfy the (0, 1) OpenEnv rule
        normalized_score = max(0.01, min(0.99, raw_total))
        
        # Determine success visually, but report the strictly bounded score
        is_success = "true" if normalized_score >= 0.7 else "false"
        
        rewards_joined = ",".join(rewards_log)
        print(f"[END] success={is_success} steps={step_count} score={normalized_score:.2f} rewards={rewards_joined}")

    except Exception as e:
        print(f"[STEP] step={step_count+1} action=ERROR reward=0.00 done=true error={str(e)}")
        # Output 0.01 instead of 0.00 to satisfy the strict > 0 rule on failures
        print(f"[END] success=false steps={step_count} score=0.01 rewards=0.01")

if __name__ == "__main__":
    tasks_to_run = ["easy", "medium", "hard", "clever"]
    
    for task in tasks_to_run:
        run_baseline(task)