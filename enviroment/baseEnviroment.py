# enviroment/baseEnviroment.py

class BaseEnvironment:
    def reset(self):
        raise NotImplementedError()

    def action(self, action):
        raise NotImplementedError()
