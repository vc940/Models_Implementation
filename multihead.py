import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.nn import softmax

class MultiHeadAttention(Layer):
    def __init__(self, attention_heads, train_data, test_data, epochs, batch):
        super().__init__()
        self.heads = attention_heads
        self.x_train = tf.repeat(train_data[0], repeats=self.heads, axis=-1)
        self.x_val = tf.repeat(test_data[0], repeats=self.heads, axis=-1)
        self.y_train = train_data[1]
        self.y_val = test_data[1]
        self.epochs = epochs
        self.len_embeddings = train_data[0].shape[-1]
        self.batch_size = batch
        self.optimizer = Adam()
        self.loss_fn = MeanSquaredError()

    def build(self, input_shape):
        self.multi_k = self.add_weight(
            shape=(self.len_embeddings * self.heads, self.len_embeddings),
            initializer="random_normal",
            trainable=True,
            name="wk"
        )
        self.multi_q = self.add_weight(
            shape=(self.len_embeddings * self.heads, self.len_embeddings),
            initializer="random_normal",
            trainable=True,
            name="wq"
        )
        self.multi_v = self.add_weight(
            shape=(self.len_embeddings * self.heads, self.len_embeddings),
            initializer="random_normal",
            trainable=True,
            name="wv"
        )

    def attention_matrix(self, sequence):
        K = tf.matmul(sequence, self.multi_k)  # Keys
        Q = tf.matmul(sequence, self.multi_q)  # Queries
        V = tf.matmul(sequence, self.multi_v)  # Values

        # Compute attention weights
        attention = softmax(tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.len_embeddings)), axis=-1)
        return attention, V

    def contextual_embeddings(self, sequence):
        attention_weights, V = self.attention_matrix(sequence)
        return tf.matmul(attention_weights, V)

    def trainloop(self):
        for epoch in range(self.epochs):
            train_loss = 0
            accumulated_gradients = [tf.zeros_like(w) for w in self.trainable_weights]

            for idx, sequence in enumerate(self.x_train):
                with tf.GradientTape() as tape:
                    y_pred = self.contextual_embeddings(sequence)
                    loss = self.loss_fn(self.y_train[idx], y_pred)

                gradients = tape.gradient(loss, self.trainable_weights)
                accumulated_gradients = [
                    acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
                ]
                if (idx + 1) % self.batch_size == 0 or idx == len(self.x_train) - 1:
                    self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_weights))
                    accumulated_gradients = [tf.zeros_like(w) for w in self.trainable_weights]

                train_loss += loss.numpy()

            validation_loss = 0
            for idx, sequence in enumerate(self.x_val):
                y_pred_val = self.contextual_embeddings(sequence)
                validation_loss += self.loss_fn(self.y_val[idx], y_pred_val).numpy()

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
