import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.nn import softmax

class Attention(Layer):
    def __init__(self, embedding_length=512, max_length=7, training_data=(None, None), validation_data=(None, None), epochs=10):
        super(Attention, self).__init__()
        self.d_w = embedding_length
        self.max_len = max_length
        self.epochs = epochs
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.optimizer = Adam()

    def build(self, input_shape):
        # Initialize weights
        self.wk = self.add_weight(shape=(self.d_w, self.d_w), initializer="random_normal", trainable=True, name="wk")
        self.bk = self.add_weight(shape=(self.d_w,), initializer="random_normal", trainable=True, name="bk")
        self.wq = self.add_weight(shape=(self.d_w, self.d_w), initializer="random_normal", trainable=True, name="wq")
        self.bq = self.add_weight(shape=(self.d_w,), initializer="random_normal", trainable=True, name="bq")
        self.wv = self.add_weight(shape=(self.d_w, self.d_w), initializer="random_normal", trainable=True, name="wv")
        self.bv = self.add_weight(shape=(self.d_w,), initializer="random_normal", trainable=True, name="bv")

    def attention_matrix(self, train_data):
        K = tf.matmul(train_data, self.wk) + self.bk
        Q = tf.matmul(train_data, self.wq) + self.bq
        V = tf.matmul(train_data, self.wv) + self.bv
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_w, tf.float32))
        attention_weights = softmax(scores, axis=-1)
        return attention_weights, V

    def contextual_embeddings(self, sequence):
        attention_weights, V = self.attention_matrix(sequence)
        return tf.matmul(attention_weights, V)

    def call(self, inputs):
        return self.contextual_embeddings(inputs)

    def train_step(self):
        loss_fn = MeanSquaredError()
        for epoch in range(self.epochs):
            train_loss = 0
            for idx, sequence in enumerate(self.x_train):
                with tf.GradientTape() as tape:
                    y_pred = self.contextual_embeddings(sequence)
                    loss = loss_fn(self.y_train[idx], y_pred)
                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                train_loss += loss.numpy()
            validation_loss = 0
            for idx, sequence in enumerate(self.x_val):
                y_pred_val = self.contextual_embeddings(sequence)
                validation_loss += loss_fn(self.y_val[idx], y_pred_val).numpy()

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
