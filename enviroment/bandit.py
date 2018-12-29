# enviroment/bandit.py

from enviroment.baseEnviroment import BaseEnvironment
import random

class BanditEnvironment(BaseEnvironment):
    def __init__(self, arm_config=[0.1, 0.9], is_random=False):
        assert all(0.0 <= a <= 1.0 for a in arm_config)

        if is_random:
            self.arm_config = arm_config[::-1] if random.random() < 0.5 else arm_config
        else:
            self.arm_config = arm_config

        self.index_bad_arm = self.arm_config.index(min(self.arm_config))

    def reset(self):
        pass

    def action(self, action):
        reward_prob = self.arm_config[action]

        if reward_prob > random.random():
            return 1.0
        else:
            return 0.0
