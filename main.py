from enviroment.bandit import BanditEnvironment
from models.lookup import LookupTable
from models.simpleLSTM import MetaLearner

from utils import testing
import json

import tensorflow as tf
import numpy as np
np.random.seed(48)
tf.random.set_random_seed(48)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
random.seed(48)

def cal_discounted_reward(rewards, gamma=0.96):
    discounted_reward = []
    cumulative_sum = 0
    for i, r in enumerate(reversed(rewards)):
        cumulative_sum = (cumulative_sum + r)*gamma
        discounted_reward.append(cumulative_sum)
    return discounted_reward[::-1]

def experiment(config, reporter):
    total_episodes = 1500
    game_length = 100
    calculate_avg_reward = 10
    bandit_arm_configuration = [0.3, 0.7]

    total_reward_list, total_regret_list = [], []
    sum_reward, total_regret = 0, 0
    regret_overtime = []

    agent = MetaLearner(
        number_actions=2,
        units=config["unit"]
    )
    observation, action_taken, reward = [], [], []

    tf.enable_eager_execution()
    count = 0

    global_step = tf.train.get_or_create_global_step()
    experiment_name = "Learning2RLLSTM"
    folder_name = experiment_name + "_" + str(config["lr"]) + "_" + str(config["unit"])
    summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + folder_name + "/learn")

    optimizer = tf.train.RMSPropOptimizer(learning_rate=config["lr"])
    saver = tf.train.Checkpoint(optimizer=optimizer,
                            model=agent,
                            optimizer_step=tf.train.get_or_create_global_step())


    for episode in range(total_episodes):
        if episode%1 == 0:
            bandit = BanditEnvironment(arm_config=bandit_arm_configuration, is_random=True)

        current_observation = [[0.0, 0.0, 0.0, 0.0]]
        state = agent.reset_state(1)

        for i in range(game_length):
            current_observation = tf.convert_to_tensor(current_observation)

            numpy_policy, _, _, state = agent(current_observation, state)
            numpy_policy = numpy_policy.numpy()

            choice = np.random.choice(2, p=numpy_policy[0])

            action_onehot = [0.0, 0.0]
            action_onehot[choice] = 1.0

            if choice == bandit.index_bad_arm:
                total_regret += 1

            r = bandit.action(choice)
            reward.append(r)
            action_taken.append(action_onehot)

            observation.append(current_observation)
            current_observation = [[(i+1.0)/game_length, r] + action_onehot]

            sum_reward += r

        global_step.assign_add(1)
        count += 1
        discounted_reward = cal_discounted_reward(reward)
        action_taken_numpy, dis_reward_numpy = np.array(action_taken), np.array(discounted_reward)

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            if episode%calculate_avg_reward == 0 and episode > 0:
                avg_reward = sum(total_reward_list)/calculate_avg_reward
                avg_total_regret = sum(total_regret_list)/calculate_avg_reward

                total_reward_list , total_regret_list= [], []
                tf.contrib.summary.scalar('avg_reward', avg_reward)
                tf.contrib.summary.scalar('avg_regret', avg_total_regret)
                both_exp = testing(agent, bandit_arm_configuration, folder_name) + testing(agent, bandit_arm_configuration[::-1], folder_name)
                tf.contrib.summary.scalar('total_regret', both_exp)
                reporter(total_regret=-both_exp, timesteps_total=count, average_reward=avg_reward, average_regret=avg_total_regret)

        total_reward_list.append(sum_reward)
        total_regret_list.append(total_regret)
        regret_overtime.append(total_regret)

        with tf.GradientTape() as tape:
            tape.watch(agent.variables)

            total_loss = 0
            state = agent.reset_state(1)
            for i in range(game_length):
                policy_prob, policy, value, state = agent(
                    tf.convert_to_tensor(observation[i]),
                    state
                )
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=policy,
                    labels=tf.convert_to_tensor(action_taken_numpy[i])
                )

                value_loss = tf.losses.mean_squared_error(
                    tf.convert_to_tensor([[discounted_reward[i]]]),
                    value
                )

                entropy_loss = tf.reduce_sum(policy_prob * tf.math.log(policy_prob))

                total_loss += loss * (discounted_reward[i] - value) - entropy_loss * 0.01 + value_loss*0.5

            total_loss /= game_length

            grad = tape.gradient(total_loss, agent.variables)
            optimizer.apply_gradients(list(zip(grad, agent.variables)))

        observation, action_taken, reward = [], [], []
        sum_reward, total_regret = 0, 0

    saver.save(f'save/{folder_name}.ckpt')

    testing(agent, bandit_arm_configuration, folder_name, is_cumulative_plot=True, is_record=True, show_prob=True),

import ray
from ray.tune import register_trainable, grid_search, run_experiments
from ray import tune
ray.init()

# def any_function(**kwargs):
#     print(f"At {kwargs['timesteps_total']}")
#
# experiment({"lr": 0.0003, "unit": 48}, any_function)

run_experiments({
    "my_experiment": {
        "run": experiment,
        "stop": {"total_regret": -40},
        "config": {
            "lr": grid_search([1e-4, 3e-4, 5e-4, 7e-4, 1e-5, 3e-5, 5e-5, 7e-5]),
            # 64, 10
            "unit": grid_search([48]),
        },
    }
})
