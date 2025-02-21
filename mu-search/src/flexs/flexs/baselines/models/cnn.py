"""Define a baseline CNN Model."""
import tensorflow as tf

from . import keras_model
import numpy as np
import flexs.utils.sequence_utils as s_utils
import torch


class CNN(keras_model.KerasModel):
    """A baseline CNN model with 3 conv layers and 2 dense layers."""

    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        hidden_size: int,
        alphabet: str,
        loss="MSE",
        name: str = None,
        batch_size: int = 256,
        epochs: int = 20,
    ):
        """Create the CNN."""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    num_filters,
                    len(alphabet) - 1,
                    padding="valid",
                    strides=1,
                    input_shape=(seq_len, len(alphabet)),
                ),
                tf.keras.layers.Conv1D(
                    num_filters, 20, padding="same", activation="relu", strides=1
                ),
                tf.keras.layers.MaxPooling1D(1),
                tf.keras.layers.Conv1D(
                    num_filters,
                    len(alphabet) - 1,
                    padding="same",
                    activation="relu",
                    strides=1,
                ),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dense(hidden_size, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(loss=loss, optimizer="adam", metrics=["mse"])

        if name is None:
            name = f"CNN_hidden_size_{hidden_size}_num_filters_{num_filters}"

        super().__init__(
            model,
            alphabet=alphabet,
            name=name,
            batch_size=batch_size,
            epochs=epochs,
        )

    # @tf.function
    def gradient_function(self, sequences):
        one_hots = tf.convert_to_tensor(
            np.array(
                [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences]
            ),
            dtype=tf.float32,
        )
        
        # Use GradientTape to track operations for gradient computation
        with tf.GradientTape() as tape:
            # Tell TensorFlow to watch this tensor for gradient computation
            tape.watch(one_hots)
            # Get predictions from the model
            predictions = self.model(one_hots, training=False)
            predictions = tf.squeeze(predictions, axis=1)
        
        # Compute gradients
        gradients = tape.gradient(predictions, one_hots)
        gradients = torch.tensor(gradients.numpy())
        # tf.print(f"cnn.py 84 gradients: {gradients}, one_hots: {one_hots}")
        gradients_cur = tf.reduce_sum(gradients * one_hots, axis=-1, keepdims=True)
        delta_ij = gradients - gradients_cur
        
        return gradients, delta_ij
