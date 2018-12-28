# enviroment/bandit.py

from enviroment.baseEnviroment import BaseEnvironment
import random

class BanditEnvironment(BaseEnvironment):
    def __init__(self, arm_config=[0.1, 0.9]):
        assert all(0.0 <= a <= 1.0 for a in arm_config)
        assert sum(arm_config) == 1.0

        self.arm_config = arm_config

    def reset(self):
        pass

    def action(self, action):
        reward_prob = self.arm_config[action]

        if reward_prob > random.random():
            return 1.0
        else:
            return 0.0
