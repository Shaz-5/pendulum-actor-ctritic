import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    """

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, seed=42):
        """
        Initialize parameters.
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset internal state to mean (mu).
        """
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """
        Return internal state as a noise sample after updating it.
        """
        noise = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.size)
        self.state += noise
        return self.state