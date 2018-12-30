import tensorflow as tf
tf.enable_eager_execution()
from enviroment.bandit import BanditEnvironment
import numpy as np
np.random.seed(48)
tf.random.set_random_seed(48)

def testing(agent, arm_configuration, folder_name, is_record=False, game_length=100):
    bandit = BanditEnvironment(arm_config=arm_configuration)
    current_observation = [[0.0, 0.0, 0.0, 0.0]]
    state = agent.reset_state(1)

    cumulative_regret = 0

    if is_record:
        summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + experiment_name + "/test")
        global_step = tf.train.get_or_create_global_step()

    # with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for i in range(game_length):
        # global_step.assign_add(1)
        current_observation = tf.convert_to_tensor(current_observation)

        numpy_policy, _, _, state = agent(current_observation, state)
        numpy_policy = numpy_policy.numpy()

        # print(f"At {i} -- {numpy_policy}")
        # tf.contrib.summary.histogram('policy', numpy_policy)

        choice = np.random.choice(2, p=numpy_policy[0])
        if choice == bandit.index_bad_arm:
            cumulative_regret += 1

        # tf.contrib.summary.scalar('cumulative_regret', cumulative_regret)

        action_onehot = [0.0, 0.0]
        action_onehot[choice] = 1.0
        r = bandit.action(choice)
        current_observation = [[i+1.0, r] + action_onehot]

    return cumulative_regret
