

import os, random, math, numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.nn.functional as F

ENV_ID = "Acrobot-v1"
SEED = 0
TOTAL_STEPS = 220_000
GAMMA = 0.99
N_STEP = 3
LR = 1e-3
BATCH_SIZE = 256
REPLAY_SIZE = 200_000
START_LEARN = 5_000
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 100_000
TARGET_UPDATE_EVERY = 1_000
GRAD_CLIP = 5.0
EVAL_EVERY = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunningNorm:
    def __init__(self, shape, eps=1e-5):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.eps = eps
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        )
    def forward(self, x):
        z = self.feat(x)
        v = self.val(z)
        a = self.adv(z)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class NStepBuffer:
    """Holds last N transitions to form n-step returns before pushing to replay."""
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = deque()
    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
        if len(self.buf) < self.n:
            return None
        return self._make_nstep()
    def flush_all(self):
        """Flush remaining partial n-step transitions (when episode ends)."""
        out = []
        while len(self.buf) > 0:
            out.append(self._make_nstep())
        return out
    def _make_nstep(self):
        R = 0.0
        s, a = self.buf[0][0], self.buf[0][1]
        done_flag = 0.0
        for i in range(self.n):
            si, ai, ri, s2i, di = self.buf[i]
            R += (self.gamma ** i) * ri
            if di:
                done_flag = 1.0
                s2 = s2i
                self.buf.popleft()
                for _ in range(len(self.buf)):
                    self.buf.popleft()
                return (s, a, R, s2, done_flag)
        s2 = self.buf[self.n - 1][3]
        self.buf.popleft()
        return (s, a, R, s2, done_flag)

class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.size = 0
        self.idx = 0
        self.s  = None
        self.a  = None
        self.r  = None
        self.s2 = None
        self.d  = None
        self._init = False
    def _ensure(self, obs_dim):
        self.s  = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.a  = np.zeros((self.cap, 1), dtype=np.int64)
        self.r  = np.zeros((self.cap, 1), dtype=np.float32)
        self.s2 = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.d  = np.zeros((self.cap, 1), dtype=np.float32)
        self._init = True
    def push(self, trans, obs_dim):
        if not self._init:
            self._ensure(obs_dim)
        s,a,r,s2,d = trans
        self.s[self.idx]  = s
        self.a[self.idx,0]= a
        self.r[self.idx,0]= r
        self.s2[self.idx] = s2
        self.d[self.idx,0]= d
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    def sample(self, bs):
        idxs = np.random.randint(0, self.size, size=bs)
        return (
            torch.as_tensor(self.s[idxs],  dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.a[idxs],  dtype=torch.int64,   device=DEVICE),
            torch.as_tensor(self.r[idxs],  dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.s2[idxs], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(self.d[idxs],  dtype=torch.float32, device=DEVICE),
        )
    def __len__(self): return self.size
def epsilon_by_step(t):
    if t >= EPS_DECAY_STEPS: return EPS_END
    frac = 1.0 - (t / EPS_DECAY_STEPS)
    return EPS_END + (EPS_START - EPS_END) * frac

def train_step(online, target, optim, replay):
    s, a, r, s2, d = replay.sample(BATCH_SIZE)
    with torch.no_grad():
        next_a = online(s2).argmax(dim=1, keepdim=True)
        next_q = target(s2).gather(1, next_a)
        td_target = r + (GAMMA ** N_STEP) * (1 - d) * next_q
    q_sa = online(s).gather(1, a)
    loss = F.smooth_l1_loss(q_sa, td_target)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
    optim.step()
    return loss.item()
def make_env(seed=None, render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None: env.reset(seed=seed)
    return env

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

env = make_env(seed=SEED)
obs_dim = env.observation_space.shape[0]
n_act   = env.action_space.n

obs_rms = RunningNorm(obs_dim)

online = DuelingQNet(obs_dim, n_act).to(DEVICE)
target = DuelingQNet(obs_dim, n_act).to(DEVICE)
target.load_state_dict(online.state_dict())
optim = torch.optim.Adam(online.parameters(), lr=LR)

replay = Replay(REPLAY_SIZE)
nstep_buf = NStepBuffer(N_STEP, GAMMA)

obs, _ = env.reset(seed=SEED)
ep_ret, ep_len = 0.0, 0
global_step = 0
best_eval = -1e9

print("Training (Dueling DDQN + 3-step) on Acrobot-v1 ...")
while global_step < TOTAL_STEPS:
    obs_rms.update(obs)
    obs_norm = obs_rms.normalize(obs)

    eps = epsilon_by_step(global_step)
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            q = online(torch.as_tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            action = int(q.argmax(dim=1).item())

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    trans = nstep_buf.push(obs_norm, action, reward, obs_rms.normalize(next_obs), float(done))
    if trans is not None:
        replay.push(trans, obs_dim)

    obs = next_obs
    ep_ret += reward; ep_len += 1
    global_step += 1

    if done:
        for t in nstep_buf.flush_all():
            replay.push(t, obs_dim)
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
    if len(replay) >= START_LEARN:
        loss = train_step(online, target, optim, replay)
    if global_step % TARGET_UPDATE_EVERY == 0:
        target.load_state_dict(online.state_dict())
    if global_step % EVAL_EVERY == 0:
        eval_env = make_env(seed=SEED+1)
        returns = []
        for ep in range(5):
            eo, _ = eval_env.reset(seed=SEED+100+ep)
            R = 0.0
            for t in range(1000):
                eo_n = obs_rms.normalize(eo)
                with torch.no_grad():
                    a = int(online(torch.as_tensor(eo_n, dtype=torch.float32, device=DEVICE)
                                   .unsqueeze(0)).argmax().item())
                eo, r, term, trunc, _ = eval_env.step(a)
                R += r
                if term or trunc: break
            returns.append(R)
        mret = float(np.mean(returns))
        best_eval = max(best_eval, mret)
        print(f"[step {global_step}] eval mean return: {mret:.1f} | best: {best_eval:.1f} | eps≈{eps:.2f}")
        eval_env.close()

env.close()
print("Training done.")

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

