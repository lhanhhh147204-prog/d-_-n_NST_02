# ============================================================
# FILE: core/models/unet.py
# CHỨC NĂNG: Kiến trúc mô hình Hybrid Attention U-Net
# TÁCH TỪ: core/model_unet.py (chỉ phần kiến trúc mạng)
# ============================================================

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config.settings import IMG_SIZE, NUM_OUTPUT_CHANNELS


def conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
    skip = conv_block(x, filters, dropout=dropout)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pooled


def attention_gate(skip: tf.Tensor, gating: tf.Tensor, filters: int) -> tf.Tensor:
    theta = layers.Conv2D(filters, 1, padding="same", use_bias=False)(skip)
    phi = layers.Conv2D(filters, 1, padding="same", use_bias=False)(gating)
    act = layers.Activation("relu")(layers.Add()([theta, phi]))
    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([skip, psi])


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    skip = attention_gate(skip, x, filters)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout=dropout)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 1),
    output_channels: int = NUM_OUTPUT_CHANNELS,
    base_filters: int = 32,
    name: str = "Hybrid_Attention_UNet",
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="image")
    f = base_filters

    s1, p1 = encoder_block(inputs, f, dropout=0.00)
    s2, p2 = encoder_block(p1, f * 2, dropout=0.00)
    s3, p3 = encoder_block(p2, f * 4, dropout=0.05)
    s4, p4 = encoder_block(p3, f * 8, dropout=0.10)

    b = conv_block(p4, f * 16, dropout=0.15)

    d1 = decoder_block(b, s4, f * 8, dropout=0.10)
    d2 = decoder_block(d1, s3, f * 4, dropout=0.05)
    d3 = decoder_block(d2, s2, f * 2, dropout=0.00)
    d4 = decoder_block(d3, s1, f, dropout=0.00)

    outputs = layers.Conv2D(
        output_channels, 1, padding="same", activation="sigmoid", dtype="float32", name="mask_abce"
    )(d4)
    return keras.Model(inputs, outputs, name=name)
