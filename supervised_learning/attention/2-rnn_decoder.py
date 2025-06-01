#!/usr/bin/env python3
"""This modlue contains the RNNDcoder class"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


# class RNNDecoder(tf.keras.layers.Layer):
#     """This class represents a decoder for a transformer
#         and inherits from tf.keras.layers.Layer"""
#     def __init__(self, vocab, embedding, units, batch):
#         """RNNDecoder constructor
#             Args:
#                 vocab (int): represents the size of the target vocabulary
#                 embedding (int): represents the dimensionality of the
# embedding
#                     vector
#                 units (int): represents the number of hidden units in the
#                 RNN cell
#                 batch (int): represents the batch size
#         """
#         super(RNNDecoder, self).__init__()
#         self.batch = batch
#         self.units = units
#         # Embedding layer to convert input tokens to dense vectors
#         # of fixed size
#         self.embedding = tf.keras.layers.Embedding(vocab, embedding)

#         # GRU layer to process the embedded input sequence
#         # return_sequences=True: return the full sequence of outputs for
#         self.gru = tf.keras.layers.GRU(units,
#                                        return_sequences=True,
#                                        return_state=True,
#                                        recurrent_initializer='glorot_uniform')

#         # F is a dense layer with vocab units
#         self.F = tf.keras.layers.Dense(vocab)

#         # Self attention mechanism
#         self.attention = SelfAttention(self.units)

#     def call(self, x, s_prev, hidden_states):
#         """This method builds the decoder
#             Args:
#                 x (tf.Tensor): a tensor of shape (batch, 1) containing
#                     the input to the decoder
#                 s_prev (tf.Tensor): a tensor of shape (batch, units)
#                     containing the previous decoder hidden state
#                 hidden_states (tf.Tensor): a tensor of shape (batch,
#                     input_seq_len, units) containing the outputs of the
# encoder
#             Returns:
#                 tf.Tensor, tf.Tensor: the outputs of the decoder and the new
#                     hidden state
#         """
#         # Pass the input through the embedding layer to convert tokens
#         # to dense vectors
#         embeddings = self.embedding(x)

#         # Apply self-attention mechanism to get the context vector
#         context_vector, attention_weights = self.attention(
#             s_prev, hidden_states)

#         # Concatenate the context vector with the embedded input
#         # while reshaping the context vector
#         x = tf.concat([tf.expand_dims(context_vector, 1), embeddings],
#  axis=-1)

#         # Pass the concatenated vector through the GRU layer
#         output, s = self.gru(x, initial_state=s_prev)

#         # Reshape the output to (batch, units)
#         output = tf.reshape(output, (-1, output.shape[2]))

#         # Pass the output through the dense layer to get the final output
#         y = self.F(output)

#         return y, s

class RNNDecoder(tf.keras.layers.Layer):
    """
    This class represents the decoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNDecoder.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass for the RNN decoder with attention mechanism.

        :param x: tensor, shape `(batch, 1)`, previous word in the target
        :param s_prev: tensor, shape `(batch, units)` previous decoder
            hidden state
        :param hidden_states: tensor, shape `(batch, input_seq_len, units)`
             outputs of the encoder

        :return:
        - y: tensor, shape `(batch, vocab)` output word as a one hot vector
        in the target vocabulary
        - s: tensor, shape `(batch, units)` new decoder hidden state
        """
        context, _ = self.attention(s_prev, hidden_states)

        # Pass the previous word index through the embedding layer
        x = self.embedding(x)

        # Concatenate the context vector with x
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated vector through the GRU layer
        output, s = self.gru(x)

        # Remove the extra axis
        output = tf.squeeze(output, axis=1)

        # Pass the GRU output through the Dense layer to predict the next word
        y = self.F(output)

        return y, s
