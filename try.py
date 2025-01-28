import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import softmax
from tensorflow.keras.losses import MeanSquaredError
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

a = tf.Variable([[1,2,4],[3,4,6]])
print(tf.repeat(a,repeats = 2 ,axis =1))