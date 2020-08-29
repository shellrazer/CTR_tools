import tensorflow as tf
from tensorflow.keras import layers, constraints

# Additive
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, value):
        # value (usually encoder output) shape == (batch_size, sequence_len, embedding_dim)
        # query shape == (batch_size, embedding_dim)
        
        # hidden_with_time_axis shape == (batch_size, 1, embedding_dim)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, sequence_len, embedding_dim)
        score = tf.nn.tanh(self.W1(value) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, sequence_len, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# Baswani
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ScaledDotProductAttention, self).__init__()
        self.units = units
        self.square_root = tf.math.sqrt(tf.cast(self.units,tf.float32))

    def call(self, query, value):
        
        # query must be a hidden vector of shape=(None, hidden_size), 
        # while value has shape=(None, steps, hidden_size)

        # expand the dimension of the query, (None, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, axis=1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are performing the dot product along that dim
        score = tf.matmul(a=query_with_time_axis, b=value, transpose_b=True)
        score = tf.divide(score, self.square_root)
        score = tf.transpose(score, perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Luong
class GeneralAttention(tf.keras.layers.Layer):
    def __init__(self, units, kernel_mn=2):
        super(GeneralAttention, self).__init__()
        self.units = units
        self.kernel_mn = kernel_mn
        self.score_weight = tf.keras.Dense(units, kernel_constraint=tf.keras.max_norm(kernel_mn))  

    def call(self, query, value):
        
        # query must be a hidden vector of shape=(None, hidden_size), 
        # while value has shape=(None, steps, hidden_size)

        # expand the dimension of the query, (None, 1, hidden_size)
        target_with_time_axis = tf.expand_dims(query, axis=1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are performing the dot product along that dim
        score = tf.matmul(a=target_with_time_axis, b=self.score_weight(value), transpose_b=True)
        score = tf.transpose(score, perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
