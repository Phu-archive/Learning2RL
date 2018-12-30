import json
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
from os.path import expanduser

class ExperimentConfiguration:
    def __init__(self, **kwargs):
        self.data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '-'.join(str(k) + ":" + str(v) for k, v in self.data.items())

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)

total_regret_data = {}
average_reward_data = {}
average_regret_data = {}

def show_value(path='ray_results/my_experiment', N=10, plot_variable="total_regret"):
    home = expanduser("~")
    path = os.path.join(home, path)
    avg_regret_test = []
    for i, p in enumerate(os.listdir(path)):
        if not "experiment" in p:
            continue
        file_path = os.path.join(path, p)
        list_tracker = []
        with open(os.path.join(file_path, 'result.json')) as f:
            list_tracker = [json.loads(lines) for lines in f.readlines()]

        with open(os.path.join(file_path, 'params.json')) as f:
            config = json.load(f)
            config = ExperimentConfiguration.from_json(config)
            display_config = str(config)

        if not display_config in ["lr:3e-05-unit:48", "lr:0.0007-unit:48", "lr:1e-05-unit:48", "lr:0.0001-unit:10"]:
            continue

        total_regret,\
        average_reward,\
        average_regret, \
        timestep = map(list, zip(*map(lambda x: (x['total_regret'],
                                    x['average_reward'], x['average_regret'], x['timesteps_this_iter']),list_tracker)))

        # Smoothing using window avg
        total_regret_data[display_config] = np.convolve(total_regret, np.ones((N,))/N, mode='valid')
        average_reward_data[display_config] = np.convolve(average_reward, np.ones((N,))/N, mode='valid')
        average_regret_data[display_config] = np.convolve(average_regret, np.ones((N,))/N, mode='valid')

        avg_regret_test.append((display_config, np.mean(total_regret)))

    total_regret_df = pd.DataFrame(total_regret_data)
    average_reward_df = pd.DataFrame(average_reward_data)
    average_regret_df = pd.DataFrame(average_regret_data)

    switch={
        "total_regret": total_regret_df,
        "average_reward": average_reward_df,
        "average_regret": average_regret_df
    }

    plt.figure(figsize=(15,5))
    ax = sns.lineplot(
        data=switch[plot_variable],
        dashes=False
    )
    ax.set_title(plot_variable)
    plt.show()


# if __name__ == '__main__':
#     show_value()
