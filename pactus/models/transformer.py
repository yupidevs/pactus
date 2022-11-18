import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, ff_dim2, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv1 = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv2 = layers.Conv1D(filters=ff_dim2, kernel_size=1)
        self.supports_masking = True

    def call(self, inputs, training, mask=None):
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")

        out_norm1 = self.layernorm1(inputs, training=training)
        out_att = self.att(
            out_norm1, out_norm1, training=training, attention_mask=padding_mask
        )
        out_drop1 = self.dropout1(out_att, training=training)
        res = out_drop1 + inputs
        out_norm2 = self.layernorm2(res, training=training)
        out_conv1 = self.conv1(out_norm2, training=training)
        out_drop2 = self.dropout2(out_conv1, training=training)
        out_conv2 = self.conv2(out_drop2, training=training)
        return out_conv2 + res


def build_model(
    n_classes,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0.0,
    mlp_dropout=0.0,
    mask=None,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    _x = inputs
    if mask is not None:
        _x = layers.Masking(mask_value=mask)(_x)
    for _ in range(num_transformer_blocks):
        _x = TransformerBlock(
            head_size,
            num_heads,
            ff_dim,
            inputs.shape[-1],
            dropout,
        )(_x)

    _x = layers.GlobalAveragePooling2D(data_format="channels_first")(_x)
    for dim in mlp_units:
        _x = layers.Dense(dim, activation="relu")(_x)
        _x = layers.Dropout(mlp_dropout)(_x)
    outputs = layers.Dense(n_classes, activation="softmax")(_x)
    return keras.Model(inputs, outputs)
