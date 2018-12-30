from utils import testing

import os
from os.path import expanduser

from models.simpleLSTM import MetaLearner
import json

import tensorflow as tf
tf.enable_eager_execution()

home = expanduser("~")
save_path = 'experiment_1_lr=0.0003,unit=48_2018-12-30_16-13-40fqzpq3mt'
main_folder = os.path.join(home, 'ray_results/my_experiment', save_path)
path = os.path.join(main_folder, save_path, '/save')

with open(os.path.join(main_folder, 'params.json'), 'r') as f:
    config = json.load(f)


agent = MetaLearner(
    number_actions=2,
    units=config["unit"]
)

print("------------------------------------------------")

optimizer = tf.train.RMSPropOptimizer(learning_rate=config["lr"])
saver = tf.train.Checkpoint(optimizer=optimizer,
                        model=agent,
                        optimizer_step=tf.train.get_or_create_global_step())

print(saver.restore(tf.train.latest_checkpoint(path)))
testing(
    agent,
    [0.3, 0.7],
    None,
    show_prob=True
)
