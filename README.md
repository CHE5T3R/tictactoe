# Reinforcement Learning Tic-Tac-Toe Agent

### Course
CENG 3511: Artificial Intelligence  
**Midterm Project** – Build an AI Model to Run a Game

---

## 🧩 Project Description
This project implements a **Reinforcement Learning Agent** that learns to play **Tic-Tac-Toe** using a Q-learning approach.  
The AI can be trained to play as either **X** or **O** and can later play interactively against a human player.

---

## 🧠 Method
- **Algorithm:** Q-Learning (Tabular)
- **State Representation:** 9-cell vector (values: `1` for X, `-1` for O, `0` for empty)
- **Action:** Selecting an available cell (0–8)
- **Reward Function:**
  - Win → +1
  - Lose → −1
  - Draw → 0
- **Discount Factor (γ):** 0.6  
- **Exploration Rate (ε):** Starts high (0.9) and decays gradually

---

## ⚙️ Requirements
- Python 3.8+
- `numpy`
- (optional) `matplotlib` for visualization

Install dependencies:
```bash
pip install numpy matplotlib
