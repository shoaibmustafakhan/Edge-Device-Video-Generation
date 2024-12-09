# model.py
from keras.layers import Conv3D, ConvLSTM2D, Input, TimeDistributed, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os

class PredRNN:
    def __init__(self):
        self.checkpoint_dir = 'checkpoints'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape, name='input_layer')

        x = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu"
        )(inputs)

        x = Conv3D(
            filters=64,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu"
        )(x)

        outputs = Conv3D(
            filters=1,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="sigmoid"
        )(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        return model

    def get_callbacks(self):
        # Changed file extension from .h5 to .keras
        checkpoint_path = os.path.join(self.checkpoint_dir, 'model_epoch_{epoch:02d}_loss_{loss:.4f}.keras')
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            save_freq='epoch'
        )
        return [checkpoint]