
import random
import DuelingQNet

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