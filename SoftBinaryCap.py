# ===============================
# SoftBinaryCap - Layer
# ===============================
# Using an untrainable [1, 0] weight layer under a 2-unit softmax layer 
# allows us to extract a single binary probability from the above convolution matrix
#################################

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