
from tensorflow.keras import layers
import tensorflow as tf

class AdditiveAttention(layers.Layer):
    def __init__(self, units=96, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W = layers.Dense(self.units, use_bias=True)
        self.v = layers.Dense(1, use_bias=False)

    def call(self, x):
        s = self.v(tf.nn.tanh(self.W(x)))   # [B,T,1]
        w = tf.nn.softmax(s, axis=1)        # [B,T,1]
        return tf.reduce_sum(w * x, axis=1) # [B,H]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
