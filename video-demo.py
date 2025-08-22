import os
from gymnasium.wrappers import RecordVideo
import torch

os.makedirs("videos", exist_ok=True)
rec_env = make_env(seed=SEED+999, render_mode="rgb_array")
rec_env = RecordVideo(
    rec_env, video_folder="videos", name_prefix="acrobot_ddqn_dueling_n3",
    episode_trigger=lambda e: True
)
obs, _ = rec_env.reset()
for t in range(1000):
    obs_n = obs_rms.normalize(obs)
    with torch.no_grad():
        a = int(online(torch.as_tensor(obs_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)).argmax().item())
    obs, r, term, trunc, _ = rec_env.step(a)
    if term or trunc: break
rec_env.close()

print("Saved demo into videos/ (acrobot_ddqn_dueling_n3-*.mp4)")