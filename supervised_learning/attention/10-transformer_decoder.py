#!/usr/bin/env python3
"""This module contains the Decoder class
    that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


# class Decoder(tf.keras.layers.Layer):
#     """This class creates a decoder for a transformer"""
#     def __init__(
#             self, N, dm, h, hidden, target_vocab, max_seq_len,
# drop_rate=0.1):
#         """Class constructor
#             Args:
#                 N (int): the number of blocks in the encoder
#                 dm (int): the dimensionality of the model
#                 h (int): the number of heads
#                 hidden (int): the number of hidden units in the fully
#                 connected layer
#                 target_vocab (int): the size of the target vocabulary
#                 max_seq_len (int): the maximum sequence length possible
#                 drop_rate (float): the dropout rate
#         """
#         # set the number of blocks
#         self.N = N
#         # Set the dementionality of the model
#         self.dm = dm
#         # Set the embedding layer
#         self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
#         # Set the positional encoding layer
#         self.positional_encoding = positional_encoding(max_seq_len, dm)
#         # Set the list of decoder blocks
#         self.blocks = [DecoderBlock(
#             dm, h, hidden, drop_rate) for _ in range(N)]
#         # Set the dropout layer
#         self.dropout = tf.keras.layers.Dropout(drop_rate)

#     # def call(self, x, encoder_output, training, look_ahead_mask,
#     # padding_mask):
#     #     """This method calls the decoder
#     #         Args:
#     #             x (tf.Tensor): contains the input to the decoder
#     #             encoder_output (tf.Tensor): contains the output of the
#     # encoder
#     #             training (bool): determines if the model is in training
#     #             look_ahead_mask (tf.Tensor): contains the
# mask to be applied
#     # to
#     #             the first multi head attention layer
#     #             padding_mask (tf.Tensor): contains the mask to
# be applied to
#     #             the second multi head attention layer
#     #         Returns:
#     #             (tf.Tensor): contains the decoder's output
#     #     """
#     #     seq_len = x.shape[1]

#     #     # apply the embedding layer
#     #     x = self.embedding(x)
#     #     # scale the embedding by the square root of the dimension
#     #     x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
#     #     # add the positional encoding
#     #     x += self.positional_encoding[:seq_len, :]

#     #     # apply the dropout layer
#     #     x = self.dropout(x, training=training)

#     #     for block in self.blocks:
#     #         x = block(
#     #             x, encoder_output, training, look_ahead_mask, padding_mask)

#     #     return x
#     def call(self, x, encoder_output, training, look_ahead_mask,
# padding_mask):
#         """function that builds a Decoder
#             Args:
#                 x is a tensor of shape (batch, target_seq_len, dm)containing
#                 the input to the decoder
#                 encoder_output is a tensor of shape (batch,
# input_seq_len, dm)
#                 containing the output of the encoder
#                 training is a boolean to determine if the model is training
#                 look_ahead_mask is the mask to be applied to the first multi
#                 head attention layer
#                 padding_mask is the mask to be applied to the second multi
#                 head attention layer
#             Returns: a tensor of shape (batch, target_seq_len, dm) containing
#             the decoder output
#         """

#         # seq_len = tf.shape(x)[1]
#         seq_len = x.shape[1]

#         # Compute the embeddings; shape (batch_size, input_seq_len, dm)
#         embeddings = self.embedding(x)
#         # Scale the embeddings
#         embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
#         # Sum the positional encodings with the embeddings
#         embeddings += self.positional_encoding[:seq_len, :]
#         # Pass the embeddings on to the dropout layer
#         output = self.dropout(embeddings, training=training)

#         # Pass the output on to the N encoder blocks (one by one)
#         for i in range(self.N):
#             output = self.blocks[i](output, encoder_output, training,
#                                     look_ahead_mask, padding_mask)

#         # Note: shape of all output tensors (batch_size, input_seq_len, dm)

#         return output


class Decoder(tf.keras.layers.Layer):
    """
    This class represents an transformer's Decoder.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Decoder.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the `Decoder`.
        """
        input_seq_len = x.shape[1]

        # embedding; new shape: (batch, input_seq_len, dm)
        x = self.embedding(x)

        # positional encoding, scaled by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        # Apply dropout to the positional encoding
        x = self.dropout(x, training=training)

        # Pass the input through each decoder block
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)

        return x
