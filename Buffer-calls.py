class NStepBuffer:
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