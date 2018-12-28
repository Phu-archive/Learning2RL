import tensorflow as tf
tf.enable_eager_execution()

class LookupTable:
    def __init__(self):
        self.variables = [tf.Variable(tf.convert_to_tensor([[0.0, 0.0]]))]

    def policy(self, num_batch):
        return tf.tile(tf.nn.softmax(self.variables[0]), [num_batch, 1]), tf.tile(self.variables[0], [num_batch, 1])
