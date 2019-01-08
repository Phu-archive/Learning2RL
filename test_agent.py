from utils import testing

import os
from os.path import expanduser

from models.simpleLSTM import MetaLearner
import json

import tensorflow as tf
tf.enable_eager_execution()

# home = expanduser("~")
save_path = 'save/'
# main_folder = os.path.join(home, 'ray_results/my_experiment', save_path)
# path = os.path.join(main_folder, save_path, '/save')

# with open(os.path.join(main_folder, 'params.json'), 'r') as f:
#     config = json.load(f)


agent = MetaLearner(
    number_actions=2,
    units=48
)

print("------------------------------------------------")

optimizer = tf.train.AdamOptimizer(learning_rate=8e-4)
saver = tf.train.Checkpoint(optimizer=optimizer,
                        model=agent,
                        optimizer_step=tf.train.get_or_create_global_step())

print(saver.restore(tf.train.latest_checkpoint(save_path)))
print(testing(
    agent,
    [0.1, 0.9],
    None,
    show_prob=True
))

print(testing(
    agent,
    [0.7, 0.3],
    None,
    show_prob=True
))
