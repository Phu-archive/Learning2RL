import tensorflow as tf

import ray
import ray.tune as tune

ray.init()

import gym
import numpy as np

gamma = 0.99

class Policy(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.fc1 = tf.keras.layers.Dense(128, use_bias=True)
    self.fc2 = tf.keras.layers.Dense(10, use_bias=True)
    self.fc3 = tf.keras.layers.Dense(2, use_bias=True)
    self.fc_value = tf.keras.layers.Dense(1, use_bias=True)

  def call(self, x):
    x = tf.nn.relu(self.fc1(x))
    x_out = tf.nn.relu(self.fc2(x))
    x = self.fc3(x_out)

    val = self.fc_value(x_out)

    return x, tf.nn.softmax(x), val

def cal_discounted_reward(rewards):
    discounted_reward = []
    cumulative_sum = 0
    for i, r in enumerate(reversed(rewards)):
        cumulative_sum = (cumulative_sum + r)*gamma
        discounted_reward.append(cumulative_sum)
    return discounted_reward[::-1]

@ray.remote
def train_fuction(config, reporter):
    tf.enable_eager_execution()
    print(f"Running with config {config}")
    env = gym.make('CartPole-v1')
    obs = env.reset()
    agent = Policy()

    observation, action_taken, reward = [], [], []
    sum_reward = 0

    eps_num = 1

    optimizers = tf.train.RMSPropOptimizer(learning_rate=config["lr"])
    while True:
        _, out, _ = agent(tf.expand_dims(tf.convert_to_tensor(obs), 0))
        choice = np.random.choice(2, p=out.numpy()[0])
        observation.append(obs)

        if choice == 0:
            action_taken.append([1.0, 0.0])
        else:
            action_taken.append([0.0, 1.0])

        obs, r, done, _ = env.step(choice)
        reward.append(r)

        sum_reward += r

        grad = None
        if done:
            print(f"At Eps: {eps_num} reward is {sum_reward}")

            discounted_rewards = cal_discounted_reward(reward)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            observation_numpy, action_taken_numpy, reward_numpy = np.array(observation), np.array(action_taken), np.array(reward)

            with tf.GradientTape() as tape:
                tape.watch(agent.variables)

                total_loss = 0

                for i in range(len(observation)-1):
                    fin_layer, pred_prob, value = agent(tf.convert_to_tensor([observation[i]]))
                    _, _, value_next = agent(tf.convert_to_tensor([observation[i+1]]))

                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=fin_layer,
                        labels=tf.convert_to_tensor([action_taken_numpy[i]])
                    )


                    value_loss = tf.losses.mean_squared_error(
                        # reward[i] + 0.99 * tf.convert_to_tensor(value_next.numpy()),
                        tf.convert_to_tensor([[discounted_rewards[i]]]),
                        value
                    )

                    entropy_loss = tf.reduce_sum(pred_prob * tf.math.log(pred_prob))

                    total_loss += -loss * (discounted_rewards[i] - value) + tf.cast(value_loss, tf.float64) * 0.5 - entropy_loss * 0.01 

                total_loss /= len(observation)

                # fin_layer, _, value = agent(tf.convert_to_tensor(observation_numpy))
                # total_loss =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=fin_layer, labels=tf.convert_to_tensor(action_taken_numpy))
                # total_loss = tf.reduce_mean(tf.multiply(loss, discounted_rewards))
                grad = tape.gradient(total_loss, agent.variables)

            optimizers.apply_gradients(zip(grad, agent.variables))

            observation, action_taken, reward = [], [], []
            obv = env.reset()
            grad = None

            if eps_num == 800:
                break
            else:
                sum_reward = 0
            eps_num += 1

    # reporter(final_sum=sum_reward)
    print(f"Done!! {config} with {sum_reward}")
    return sum_reward

# possible_lr = [n*1e-3 for n in range(1, 10)] + [n*1e-4 for n in range(1, 10)] + [n*1e-5 for n in range(1, 10)]
# a = [train_fuction.remote({"lr": i}, print) for i in possible_lr]
# total = list(zip(ray.get(a), possible_lr))
#
# print(sorted(total, key=lambda x: x[0])[::-1][:3])

ray.get(train_fuction.remote({"lr": 1e-3}, print))
