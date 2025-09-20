
from tensorflow import keras
from tensorflow.keras import layers
from .attention import AdditiveAttention

def build_seq_attn_model(timesteps, seq_feats, aux_feats,
                         num_classes=3, rnn_units=96, attn_units=96,
                         dense_units=128, dropout=0.2):
    inp_seq = keras.Input(shape=(timesteps, seq_feats), name="seq")
    x = layers.Masking()(inp_seq)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = AdditiveAttention(units=attn_units)(x)
    x = layers.Dropout(dropout)(x)

    inp_aux = keras.Input(shape=(aux_feats,), name="aux")
    a = layers.LayerNormalization()(inp_aux)

    z = layers.Concatenate()([x, a])
    z = layers.Dense(dense_units, activation="relu")(z)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(num_classes, activation="softmax")(z)

    model = keras.Model(inputs={"seq": inp_seq, "aux": inp_aux}, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
