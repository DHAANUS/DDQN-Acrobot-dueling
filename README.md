# Double Deep Q-Network (DDQN) — CartPole-v1 (PyTorch from Scratch)

This repository contains a from-scratch implementation of the **Double Deep Q-Network (DDQN)** algorithm in PyTorch.  
It extends the standard DQN by decoupling action selection and action evaluation, which helps reduce overestimation bias.  
The agent is trained on the **CartPole-v1** environment from Gymnasium.

---

## Key Features
- Fully implemented from scratch in PyTorch  
- Double Q-Learning update: separate networks for action selection and evaluation  
- Experience Replay to break correlation in training samples  
- Target Network for stable learning  
- ε-Greedy Exploration with decay  
- Logging of training rewards and demo video generation  

---

## Quickstart
Install dependencies:
```bash
pip install torch gymnasium numpy tqdm imageio imageio-ffmpeg
