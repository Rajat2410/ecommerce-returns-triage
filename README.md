---
title: "OpenEnv: E-Commerce Returns & Refund Triage"
emoji: "🛒"
colorFrom: "blue"
colorTo: "green"
sdk: "docker"
pinned: false
---

# 🛒 OpenEnv: E-Commerce Returns & Refund Triage

A **real-world OpenEnv benchmark** simulating multi-step customer support workflows for e-commerce returns and refunds.

This environment evaluates an AI agent’s ability to:

* apply strict policy rules
* gather missing information
* compute exact refund amounts
* make business trade-offs

---

# 🌟 Motivation

Customer support automation fails at the intersection of:

* **Policy compliance** — rules must be enforced strictly
* **Sequential reasoning** — conversations require multiple steps
* **Numerical accuracy** — refunds must be exact

This benchmark provides a **deterministic, reproducible environment** to evaluate all three together.

---

# 🎯 What the Agent Must Do

An agent interacting with this environment must:

1. Interpret policy rules (eligibility, timelines)
2. Ask for missing information (e.g., request evidence)
3. Choose correct actions (approve, deny, escalate)
4. Compute exact refund values (including deductions)

---

# 🏗️ Environment Architecture

```id="arch1"
.
├── server/
│   ├── app.py
│   ├── environment.py
│   └── graders.py
├── tasks/
│   ├── easy_01.json
│   ├── medium_01.json
│   ├── hard_01.json
│   └── clever_01.json
├── models.py
├── inference.py
├── openenv.yaml
├── Dockerfile
└── README.md
```

---

# 🧠 Environment Interface

## Observation Space

```json id="obs1"
{
  "ticket_id": "string",
  "customer_message": "string",
  "order_date": "YYYY-MM-DD",
  "current_date": "YYYY-MM-DD",
  "items": [
    {
      "item_id": "string",
      "name": "string",
      "price": 100.0,
      "category": "string",
      "condition": "string"
    }
  ],
  "policy_snippet": "string",
  "conversation_history": []
}
```

---

## Action Space

* `APPROVE_ELIGIBLE`
* `DENY_INELIGIBLE`
* `ASK_QUESTION`
* `ISSUE_REFUND(refund_amount: float)`
* `NO_RETURN_REFUND`

---

# 🔄 Example Interaction

### Reset

**POST /reset**

```json id="ex1"
{
  "observation": {
    "customer_message": "I opened this smartwatch...",
    "policy_snippet": "15% restocking fee..."
  }
}
```

---

### Step

**Agent Action**

```json id="ex2"
{
  "action_type": "ISSUE_REFUND",
  "refund_amount": 170.0
}
```

**Environment Response**

```json id="ex3"
{
  "observation": {...},
  "reward": 1.0,
  "done": true,
  "info": {}
}
```

---

# 🧪 Tasks

### 🟢 Easy — Eligibility Check

* Single-step decision
* Policy interpretation

---

### 🟡 Medium — Information Gathering

* Multi-step interaction
* Requires asking questions

---

### 🔴 Hard — Refund Calculation

* Applies policy + computes refund
* Includes restocking fees

---

### 🧠 Clever — Business Optimization

* Decides between return vs no-return refund
* Optimizes cost vs policy

---

# ⚖️ Reward & Grading

All graders are **deterministic (non-LLM)**.

### Reward signals:

* **+0.1** → valid intermediate step
* **-0.5** → policy violation
* **+1.0** → correct final resolution

### Example:

```id="math1"
$200 item with 15% restocking fee → refund = $170.00
```

### Final Score:

* Normalized to **[0.0, 1.0]**
* Fully reproducible

---

# 📊 Baseline Results

| Task   | Score |
| ------ | ----- |
| Easy   | 0.82  |
| Medium | 0.61  |
| Hard   | 0.47  |
| Clever | 0.44  |

---

# 💡 Novelty

This benchmark uniquely combines:

* Rule-based policy enforcement
* Multi-step conversational reasoning
* Exact numerical computation

> It evaluates reasoning, interaction, and arithmetic precision in a real-world workflow.

---

# 🚀 Getting Started

## Install dependencies

```bash id="cmd1"
pip install uv
uv lock
pip install .
```

---

## Run baseline

```bash id="cmd2"
export HF_TOKEN=your_api_key
export MODEL_NAME=gpt-4o-mini

python inference.py
```

---

## Validate environment

```bash id="cmd3"
openenv validate
```

---

# 🐳 Deployment (Hugging Face Spaces)

* Use **Docker SDK**
* Upload all files
* Add `HF_TOKEN` as a secret
* App runs on **port 7860**

---

# 🛠️ Infrastructure Constraints

* vCPU: 2
* Memory: 8GB
* Runtime: < 20 minutes

---

# 👥 Team

A team of undergraduate engineers from IIT Guwahati building real-world evaluation environments for AI agents.

* Rajat Jain
* Yash Kumar
* Shaurya Babel

---

# 🟢 Final Note

This benchmark reflects real-world customer support systems where incorrect reasoning directly impacts business cost and user trust.
