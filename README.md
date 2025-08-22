# Double Deep Q-Network (DDQN) + Dueling — Acrobot-v1 (PyTorch from Scratch)

This repository contains a from-scratch implementation of the **Double Deep Q-Network (DDQN)** algorithm in PyTorch,  
extended with a **Dueling Network architecture** for better state-value and advantage estimation.  

It is trained on the **Acrobot-v1** environment from Gymnasium, demonstrating improved stability and reduced overestimation bias compared to standard DQN.
---

## Key Features
- Fully implemented from scratch in PyTorch
- Double Q-Learning update: decoupled action selection and evaluation to reduce overestimation
- Dueling Network Architecture for separate value & advantage streams
- Experience Replay to break correlation in training samples
- Target Network for stable bootstrapping
- ε-Greedy Exploration with decay
- Demo MP4 video recording of trained agent

---

## Quickstart
Install dependencies:
```bash
pip install torch gymnasium numpy tqdm imageio imageio-ffmpeg
