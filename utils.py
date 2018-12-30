import tensorflow as tf
tf.enable_eager_execution()
from enviroment.bandit import BanditEnvironment
import numpy as np
np.random.seed(48)
tf.random.set_random_seed(48)

def testing(agent, arm_configuration, folder_name, is_record=False, game_length=100, is_cumulative_plot=False, show_prob=False):
    if is_record:
        summary_writer = tf.contrib.summary.create_file_writer("tmp/learning2RL/" + folder_name + "/test")
        global_step = tf.train.get_or_create_global_step()

    def run(arm_configuration, num_run=1):
        current_observation = [[0.0, 0.0, 0.0, 0.0]]
        state = agent.reset_state(1)
        bandit = BanditEnvironment(arm_config=arm_configuration)

        cumulative_regret = 0
        cumulative_regret_list = []

        for i in range(game_length):
            current_observation = tf.convert_to_tensor(current_observation)

            numpy_policy, _, value, state = agent(current_observation, state)
            numpy_policy = numpy_policy.numpy()

            if show_prob:
                print(numpy_policy)

            choice = np.random.choice(2, p=numpy_policy[0])
            if choice == bandit.index_bad_arm:
                cumulative_regret += 1

            cumulative_regret_list.append(cumulative_regret)

            if is_record:
                global_step.assign_add(1)
                tf.contrib.summary.scalar('cumulative_regret' + str(num_run), cumulative_regret)
                tf.contrib.summary.histogram('policy'+ str(num_run), numpy_policy)

            action_onehot = [0.0, 0.0]
            action_onehot[choice] = 1.0
            r = bandit.action(choice)
            current_observation = [[(i+1.0)/game_length, r] + action_onehot]

        return cumulative_regret, cumulative_regret_list


    if is_record:
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            cumulative_regret, cumulative_regret_list = run(arm_configuration, num_run=1)
            arm_configuration = arm_configuration[::-1]
            global_step.assign(0)
            cumulative_regret, cumulative_regret_list = run(arm_configuration, num_run=2)
    else:
        cumulative_regret, cumulative_regret_list = run(arm_configuration)

    if is_cumulative_plot:
        return {"total_regret": cumulative_regret, "regret_evo": cumulative_regret_list}
    return cumulative_regret
