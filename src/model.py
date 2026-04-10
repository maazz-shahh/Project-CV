import tensorflow as tf
from tensorflow.keras import layers, models


def baseline_autoencoder(input_shape=(128, 128, 1)):
    """
    Gray → RGB encoder–decoder with skip connections; upsample + conv decoder.
    """
    inp = layers.Input(shape=input_shape, name="gray_input")

    def enc_down(x, channels):
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(channels, 3, strides=2, padding="same", activation="relu")(x)
        return x

    x = enc_down(inp, 48)
    sk0 = x
    x = enc_down(x, 96)
    sk1 = x
    x = enc_down(x, 192)
    sk2 = x
    x = enc_down(x, 256)
    sk3 = x
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(sk3)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    def dec_up(x, skip, channels):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        return x

    x = dec_up(x, sk2, 192)
    x = dec_up(x, sk1, 128)
    x = dec_up(x, sk0, 96)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(3, 1, activation="sigmoid", name="rgb_output")(x)
    return models.Model(inp, out, name="baseline_autoencoder_restorer")


def combined_loss(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.75 * l1 + 0.25 * ssim
