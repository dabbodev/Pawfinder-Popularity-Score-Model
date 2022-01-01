# ===============================
# binaryFeatureDetector - Model
# ===============================
# Sub dividing a larger convolution matrix into several independent binary detection arms 
# allows us to train a single model to predict several binary classifications in parallel
#################################

class binaryFeatureDetector(keras.Model):

  def __init__(self):
    super(binaryFeatureDetector, self).__init__()
    self.dense1 = layers.Dense(128, use_bias=False, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.dense2 = layers.Dense(64, use_bias=False, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.dense3 = layers.Dense(32, use_bias=False, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.dense4 = layers.Dense(16, use_bias=False, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.dense5 = layers.Dense(8, use_bias=False, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.softmax = layers.Dense(2, use_bias=False, activation='softmax', kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
    self.cap = SoftBinaryCap()

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    x = self.dense5(x)
    x = self.softmax(x)
    return self.cap(x)