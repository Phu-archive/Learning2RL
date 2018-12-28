from enviroment.bandit import BanditEnvironment
from models.lookup import LookupTable
from models.simpleLSTM import MetaLearner

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

import random

def cal_discounted_reward(rewards, gamma=0.96):
    discounted_reward = []
    cumulative_sum = 0
    for i, r in enumerate(reversed(rewards)):
        cumulative_sum = (cumulative_sum + r)*gamma
        discounted_reward.append(cumulative_sum)
    return discounted_reward[::-1]

agent = MetaLearner(48)
observation, action_taken, reward = [], [], []

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)

total_episodes = 500
game_length = 100
calculate_avg_reward = 10

total_reward_list = []
sum_reward = 0

global_step = tf.train.get_or_create_global_step()
experiment_name = "Learning2RLLSTM"
summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + experiment_name)

for episode in range(total_episodes):
    if random.random() > 0.5:
        bandit = BanditEnvironment(arm_config=[0.3, 0.7])
    else:
        bandit = BanditEnvironment(arm_config=[0.7, 0.3])
    current_observation = [[0.0, 0.0, 0.0, 0.0]]
    state = agent.reset_state(1)

    for i in range(game_length):
        # numpy_policy, _ = agent.policy(1)
        current_observation = tf.convert_to_tensor(current_observation)

        numpy_policy, _, _, state = agent(current_observation, state)
        numpy_policy = numpy_policy.numpy()

        choice = np.random.choice(2, p=numpy_policy[0])

        action_onehot = [0.0, 0.0]
        action_onehot[choice] = 1.0

        r = bandit.action(choice)
        reward.append(r)
        action_taken.append(action_onehot)

        observation.append(current_observation)
        current_observation = [[i+1.0, r] + action_onehot]

        sum_reward += r

    global_step.assign_add(1)
    discounted_reward = cal_discounted_reward(reward)
    action_taken_numpy, dis_reward_numpy = np.array(action_taken), np.array(discounted_reward)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        if episode%calculate_avg_reward == 0 and episode > 0:
            avg_reward = sum(total_reward_list)/calculate_avg_reward
            print(f"Average Reward over {episode+1}/{total_episodes} is {avg_reward}")
            total_reward_list = []
            tf.contrib.summary.scalar('avg_reward', avg_reward)


    total_reward_list.append(sum_reward)

    with tf.GradientTape() as tape:
        tape.watch(agent.variables)

        total_loss = 0
        state = agent.reset_state(1)
        for i in range(game_length):
            # _, policy = agent.policy(1)

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
    sum_reward = 0

##### TESTING ######
bandit = BanditEnvironment(arm_config=[0.7, 0.3])
current_observation = [[0.0, 0.0, 0.0, 0.0]]
state = agent.reset_state(1)

with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for i in range(game_length):
        global_step.assign_add(1)
        # numpy_policy, _ = agent.policy(1)
        current_observation = tf.convert_to_tensor(current_observation)

        numpy_policy, _, _, state = agent(current_observation, state)
        numpy_policy = numpy_policy.numpy()

        print(f"At i -- {numpy_policy}")
        tf.contrib.summary.histogram('policy', numpy_policy)

        choice = np.random.choice(2, p=numpy_policy[0])

        action_onehot = [0.0, 0.0]
        action_onehot[choice] = 1.0
        r = bandit.action(choice)
        current_observation = [[i+1.0, r] + action_onehot]
