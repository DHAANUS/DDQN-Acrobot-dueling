import numpy as np
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