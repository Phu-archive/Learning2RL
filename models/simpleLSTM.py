import tensorflow as tf
tf.enable_eager_execution()

class MetaLearner(tf.keras.Model):
    def __init__(self, number_actions, units):
        super().__init__()

        self.input_layer = tf.keras.layers.Dense(units)
        self.lstm_core = tf.keras.layers.LSTMCell(units)
        self.policy_layer = tf.keras.layers.Dense(number_actions)
        self.value_layer = tf.keras.layers.Dense(1)

        self.units = units

    def call(self, obs, hidden):
        # inp = tf.nn.relu(self.input_layer(obs))
        out, state = self.lstm_core(obs, hidden)

        pre_policy = self.policy_layer(out)
        value = self.value_layer(out)
        return tf.nn.softmax(pre_policy), pre_policy, value, state

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
