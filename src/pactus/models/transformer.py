from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):  # type: ignore[misc]
    def __init__(
        self,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        ff_dim2: int,
        *args: Any,
        rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ff_dim2 = ff_dim2
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.conv1 = layers.Conv2D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv2 = layers.Conv2D(filters=ff_dim2, kernel_size=1)
        self.supports_masking = True

    def call(self, inputs: Any, training: Any = None, mask: Any = None) -> Any:
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")

        out_norm1 = self.layernorm1(inputs, training=training)
        out_att = self.att(out_norm1, out_norm1, training=training, attention_mask=padding_mask)
        out_drop1 = self.dropout1(out_att, training=training)
        res = out_drop1 + inputs
        out_norm2 = self.layernorm2(res, training=training)
        out_conv1 = self.conv1(out_norm2, training=training)
        out_drop2 = self.dropout2(out_conv1, training=training)
        out_conv2 = self.conv2(out_drop2, training=training)
        return out_conv2 + res

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "head_size": self.head_size,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "ff_dim2": self.ff_dim2,
                "rate": self.rate,
            }
        )
        full_config: dict[str, Any] = config
        return full_config


def build_model(
    n_classes: int,
    input_shape: tuple,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    num_transformer_blocks: int,
    mlp_units: list[int],
    dropout: float = 0.0,
    mlp_dropout: float = 0.0,
    mask: Any = None,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    _x = inputs
    if mask is not None:
        _x = layers.Masking(mask_value=mask)(_x)
    for _ in range(num_transformer_blocks):
        _x = TransformerBlock(head_size, num_heads, ff_dim, inputs.shape[-1], rate=dropout)(_x)

    _x = layers.GlobalAveragePooling2D(data_format="channels_first")(_x)
    for dim in mlp_units:
        _x = layers.Dense(dim, activation="relu")(_x)
        _x = layers.Dropout(mlp_dropout)(_x)
    outputs = layers.Dense(n_classes, activation="softmax")(_x)
    return keras.Model(inputs, outputs)
