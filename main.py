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

agent = MetaLearner(
    number_actions=2,
    units=48
)
observation, action_taken, reward = [], [], []

optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)

total_episodes = 5000
game_length = 100
calculate_avg_reward = 10
bandit_arm_configuration = [0.3, 0.7]

total_reward_list, total_regret_list = [], []
sum_reward, total_regret = 0, 0

global_step = tf.train.get_or_create_global_step()
experiment_name = "Learning2RLLSTM"
summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + experiment_name + "/learn")

for episode in range(total_episodes):
    # New bandit every 50 eps
    if episode%100 == 0:
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
        current_observation = [[i+1.0, r] + action_onehot]

        sum_reward += r

    global_step.assign_add(1)
    discounted_reward = cal_discounted_reward(reward)
    action_taken_numpy, dis_reward_numpy = np.array(action_taken), np.array(discounted_reward)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        if episode%calculate_avg_reward == 0 and episode > 0:
            avg_reward = sum(total_reward_list)/calculate_avg_reward
            avg_total_regret = sum(total_regret_list)/calculate_avg_reward
            print(f"Average Reward over {episode+1}/{total_episodes} is {avg_reward}")
            print(f"Average Total Regret {episode+1}/{total_episodes} is {avg_total_regret}")
            print("----------")
            total_reward_list , total_regret_list= [], []
            tf.contrib.summary.scalar('avg_reward', avg_reward)
            tf.contrib.summary.scalar('avg_regret', avg_total_regret)


    total_reward_list.append(sum_reward)
    total_regret_list.append(total_regret)

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

            total_loss += loss * (discounted_reward[i] - value) - entropy_loss * 0.02 + value_loss*0.5

        total_loss /= game_length

        grad = tape.gradient(total_loss, agent.variables)
        optimizer.apply_gradients(list(zip(grad, agent.variables)))

    observation, action_taken, reward = [], [], []
    sum_reward, total_regret = 0, 0

##### TESTING ######
bandit = BanditEnvironment(arm_config=bandit_arm_configuration)
current_observation = [[0.0, 0.0, 0.0, 0.0]]
state = agent.reset_state(1)

summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + experiment_name + "/test")
global_step = tf.train.get_or_create_global_step()

cumulative_regret = 0

with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for i in range(game_length):
        global_step.assign_add(1)
        current_observation = tf.convert_to_tensor(current_observation)

        numpy_policy, _, _, state = agent(current_observation, state)
        numpy_policy = numpy_policy.numpy()

        print(f"At {i} -- {numpy_policy}")
        tf.contrib.summary.histogram('policy', numpy_policy)

        choice = np.random.choice(2, p=numpy_policy[0])
        if choice == bandit.index_bad_arm:
            cumulative_regret += 1

        tf.contrib.summary.scalar('cumulative_regret', cumulative_regret)

        action_onehot = [0.0, 0.0]
        action_onehot[choice] = 1.0
        r = bandit.action(choice)
        current_observation = [[i+1.0, r] + action_onehot]
