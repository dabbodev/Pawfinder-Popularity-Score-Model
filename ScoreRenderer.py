class ScoreRenderer(Layer):
    def __init__(self):
        super(ScoreRenderer, self).__init__()

    def build(self, input_shape):
        self.a = self.add_weight(shape=(3,),
                                 constraint=tf.keras.constraints.NonNeg(),
                               trainable=True)
        
        self.b = self.add_weight(shape=(6,),
                                 constraint=tf.keras.constraints.NonNeg(),
                               trainable=True)
        
        self.c = self.add_weight(shape=(6,),
                                 constraint=tf.keras.constraints.NonNeg(),
                               trainable=True)
        
        

    def call(self, inputs): 
        l = tf.shape(inputs)[0]  
        pos = keras.backend.map_fn(lambda i: i * self.b, inputs[:,:6])
        neg = keras.backend.map_fn(lambda i: i * self.c, inputs[:,6:12])
        x = keras.backend.map_fn(lambda i:  i * self.a, inputs[:,12:])
        x = tf.reduce_sum(x, axis=1)
        x = keras.backend.reshape(x, (l, 1))
        pos = tf.reduce_sum(pos, axis=1)
        neg = tf.reduce_sum(neg, axis=1)
        pos = keras.backend.reshape(pos, (l, 1))
        neg = keras.backend.reshape(neg, (l, 1))
        x = tf.concat([x, (pos - neg)], 1)
        return x