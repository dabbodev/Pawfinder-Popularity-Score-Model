class SoftBinaryCap(Layer):
    def __init__(self):
        super(SoftBinaryCap, self).__init__()

    def build(self, input_shape):
        
        self.w = tf.Variable(initial_value=tf.convert_to_tensor([[1.],[0.]]),
            trainable=False
        )

    def call(self, inputs):
        x = tf.matmul(inputs, self.w)
        return x