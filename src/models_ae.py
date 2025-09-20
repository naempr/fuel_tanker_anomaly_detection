
from tensorflow import keras
from tensorflow.keras import layers

class LSTMAE(keras.Model):
    def __init__(self, timesteps, features, lstm_units=(64,32), dropout=0.1, **kw):
        super().__init__(**kw)
        self.enc1 = layers.LSTM(lstm_units[0], return_sequences=True)
        self.enc2 = layers.LSTM(lstm_units[1])
        self.do   = layers.Dropout(dropout)
        self.rep  = layers.RepeatVector(timesteps)
        self.dec1 = layers.LSTM(lstm_units[1], return_sequences=True)
        self.dec2 = layers.LSTM(lstm_units[0], return_sequences=True)
        self.out  = layers.TimeDistributed(layers.Dense(features))
    def call(self, x, training=False):
        h = self.enc1(x); h = self.enc2(h); h = self.do(h, training=training)
        h = self.rep(h); h = self.dec1(h); h = self.dec2(h)
        return self.out(h)

def build_ae(timesteps, features, cfg):
    return LSTMAE(timesteps, features,
                  tuple(cfg.get("lstm_units",(64,32))),
                  cfg.get("dropout",0.1))
