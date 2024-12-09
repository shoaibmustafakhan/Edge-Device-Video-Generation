import tensorflow as tf
from tensorflow.keras import layers, Model

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            strides=(1, patch_size, patch_size),
            padding="valid"
        )

    def call(self, inputs):
        x = self.projection(inputs)
        shape = tf.shape(x)
        batch_size = shape[0]
        frames = shape[1]
        h = shape[2]
        w = shape[3]
        x = tf.reshape(x, [batch_size, frames, h * w, self.embed_dim])
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class ViViT(Model):
    def __init__(
        self, input_shape, patch_size=4, embed_dim=256, num_heads=12,
        ff_dim=512, num_transformer_layers=10, dropout=0.1, future_frames=5
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        frames, h, w, c = input_shape
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.embed_dim = embed_dim
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=[1, frames, self.num_patches, embed_dim],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ]
        self.future_frames = future_frames
        self.conv2d_transpose = layers.Conv2DTranspose(
            filters=1,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs, training=False):
        x = self.patch_embed(inputs)
        x += self.pos_embed

        for block in self.transformer_blocks:
            x = block(x, training=training)

        future_frames = []
        patch_dim = tf.cast(tf.sqrt(tf.cast(self.num_patches, tf.float32)), tf.int32)
        for _ in range(self.future_frames):
            last_frame = x[:, -1]
            last_frame = tf.reshape(
                last_frame,
                [-1, patch_dim, patch_dim, self.embed_dim]
            )
            reconstructed_frame = self.conv2d_transpose(last_frame)
            future_frames.append(reconstructed_frame)

        return tf.stack(future_frames, axis=1)