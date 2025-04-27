
## Autoencoders
# Understanding Key Concepts in Autoencoders and Generative Models

## What is an Autoencoder?
An **Autoencoder** is a type of artificial neural network used to learn efficient codings of data. The goal of an autoencoder is to compress the input data into a lower-dimensional representation (encoding) and then reconstruct it (decoding) back to the original form. This process involves two main components:
- **Encoder**: Compresses the input data into a latent space representation.
- **Decoder**: Reconstructs the input data from the latent space representation.

Autoencoders are trained in an unsupervised manner and are used for tasks such as dimensionality reduction, data denoising, and feature learning.

---

## What is Latent Space?
**Latent Space** refers to the compressed representation of data in the middle layer of an autoencoder. This space captures the essential features of the input data in a lower-dimensional form. The latent space serves as a sort of "summary" of the data, where each point in the space corresponds to a specific set of features in the original input data. The structure of this space is crucial as it directly influences the quality of the data reconstruction by the decoder.

---

## What is a Bottleneck?
The **Bottleneck** is the narrowest point in an autoencoder's architecture, typically the latent space. It represents the layer with the fewest neurons, where the input data is compressed to its most essential features. The term "bottleneck" is used because, like the neck of a bottle, this layer constrains the flow of information, forcing the network to learn the most important aspects of the data. The effectiveness of the autoencoder depends on how well this bottleneck layer captures the critical features needed for accurate reconstruction.

---

## What is a Sparse Autoencoder?
A **Sparse Autoencoder** is a type of autoencoder that incorporates a sparsity constraint on the hidden layers. This means that during the training process, only a small number of neurons are allowed to be active (non-zero) at any given time. The sparsity constraint encourages the model to learn a more efficient and interpretable representation of the data by forcing it to capture the most salient features. Sparse autoencoders are particularly useful for feature selection and are often used in scenarios where the input data has a high dimensionality.

---

## What is a Convolutional Autoencoder?
A **Convolutional Autoencoder (CAE)** is a type of autoencoder that uses convolutional layers instead of fully connected layers to encode and decode the input data. This approach is particularly well-suited for image data, as convolutional layers can capture spatial hierarchies and local patterns in images more effectively. The structure of a convolutional autoencoder typically includes:
- **Convolutional Layers**: For feature extraction in the encoder.
- **Pooling Layers**: For downsampling and reducing dimensionality.
- **Deconvolutional/Transposed Convolutional Layers**: For reconstructing the image in the decoder.

Convolutional autoencoders are widely used in tasks such as image denoising, image generation, and anomaly detection in images.

---

## What is a Generative Model?
A **Generative Model** is a type of model in machine learning that learns the underlying distribution of the input data in order to generate new, similar data samples. Unlike discriminative models, which predict labels based on input data, generative models aim to understand how the data is generated, allowing them to produce new instances that resemble the original data. Examples of generative models include:
- **Variational Autoencoders (VAEs)**
- **Generative Adversarial Networks (GANs)**
- **Restricted Boltzmann Machines (RBMs)**

Generative models are used in applications like image synthesis, text generation, and data augmentation.

---

## What is a Variational Autoencoder?
A **Variational Autoencoder (VAE)** is a type of generative model that extends the basic autoencoder framework by introducing a probabilistic approach to the latent space. Instead of mapping inputs to a single point in the latent space, VAEs map inputs to a distribution (typically Gaussian). The key components of a VAE are:
- **Encoder**: Maps the input to the parameters of a probability distribution (mean and variance) in the latent space.
- **Decoder**: Samples from this distribution and reconstructs the input data.

VAEs are trained using a loss function that includes:
- **Reconstruction Loss**: Measures how well the decoder reconstructs the input data.
- **Kullback-Leibler Divergence**: Regularizes the latent space to follow a prior distribution, usually a standard normal distribution.

VAEs are widely used for generating new data samples and for creating smooth, continuous latent spaces that allow for meaningful data interpolations.

---

## What is the Kullback-Leibler Divergence?
**Kullback-Leibler Divergence (KLD)** is a measure of how one probability distribution diverges from a second, reference probability distribution. In the context of Variational Autoencoders, KLD is used to measure the difference between the learned latent distribution and a prior distribution (usually a standard normal distribution). The goal is to minimize this divergence so that the latent variables are well-structured and follow a known distribution.

The KLD is defined as:
\[
\text{KLD}(P || Q) = \sum_{i=1}^{n} P(x_i) \log\left(\frac{P(x_i)}{Q(x_i)}\right)
\]
Where:
- \(P(x_i)\) is the learned distribution.
- \(Q(x_i)\) is the prior distribution.

Minimizing the KLD term in the VAE loss function ensures that the latent space is organized and allows for smooth sampling, making the model capable of generating realistic and coherent data samples.

---



### Description
Question #0What is a “vanilla” autoencoder?A compression modelComposed of an encoder and decoderA generative modelLearns a latent space representation

Question #1What is a bottleneck?When you can no longer train your modelThe latent space representationThe compressed inputA layer that is smaller than the previous and next layers

Question #2What is a VAE?An adversarial networkA generative modelComposed of an encoder and decoderA compression model

Question #3What loss function(s) is/are used for trainingvanillaautoencoders?Mean Squared ErrorL2 NormalizationCross EntropyKullback-Leibler Divergence

Question #4What loss function(s) is/are used for training variational autoencoders?Mean Squared ErrorL2 NormalizationCross EntropyKullback-Leibler Divergence

0. "Vanilla" AutoencodermandatoryWrite a functiondef autoencoder(input_dims, hidden_layers, latent_dims):that creates an autoencoder:input_dimsis an integer containing the dimensions of the model inputhidden_layersis a list containing the number of nodes for each hidden layer in the encoder, respectivelythe hidden layers should be reversed for the decoderlatent_dimsis an integer containing the dimensions of the latent space representationReturns:encoder, decoder, autoencoderis the encoder modeldecoderis the decoder modelautois the full autoencoder modelThe autoencoder model should be compiled using adam optimization and binary cross-entropy lossAll layers should use areluactivation except for the last layer in the decoder, which should usesigmoid$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

autoencoder = __import__('0-vanilla').autoencoder

SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

encoder, decoder, auto = autoencoder(784, [128, 64], 32)
auto.fit(x_train, x_train, epochs=50,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./0-main.py
Epoch 1/50
235/235 [==============================] - 3s 10ms/step - loss: 0.2462 - val_loss: 0.1704
Epoch 2/50
235/235 [==============================] - 2s 10ms/step - loss: 0.1526 - val_loss: 0.1370
Epoch 3/50
235/235 [==============================] - 3s 11ms/step - loss: 0.1319 - val_loss: 0.1242
Epoch 4/50
235/235 [==============================] - 2s 10ms/step - loss: 0.1216 - val_loss: 0.1165
Epoch 5/50
235/235 [==============================] - 3s 11ms/step - loss: 0.1157 - val_loss: 0.1119

...

Epoch 46/50
235/235 [==============================] - 2s 11ms/step - loss: 0.0851 - val_loss: 0.0845
Epoch 47/50
235/235 [==============================] - 2s 11ms/step - loss: 0.0849 - val_loss: 0.0845
Epoch 48/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0848 - val_loss: 0.0842
Epoch 49/50
235/235 [==============================] - 3s 13ms/step - loss: 0.0847 - val_loss: 0.0842
Epoch 50/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0846 - val_loss: 0.0842
1/1 [==============================] - 0s 76ms/step
8.311438
1/1 [==============================] - 0s 80ms/stepRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/autoencodersFile:0-vanilla.pyHelp×Students who are done with "0. "Vanilla" Autoencoder"Review your work×Correction of "0. "Vanilla" Autoencoder"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

1. Sparse AutoencodermandatoryWrite a functiondef autoencoder(input_dims, hidden_layers, latent_dims, lambtha):that creates a sparse autoencoder:input_dimsis an integer containing the dimensions of the model inputhidden_layersis a list containing the number of nodes for each hidden layer in the encoder, respectivelythe hidden layers should be reversed for the decoderlatent_dimsis an integer containing the dimensions of the latent space representationlambthais the regularization parameter used for L1 regularization on the encoded outputReturns:encoder, decoder, autoencoderis the encoder modeldecoderis the decoder modelautois the sparse autoencoder modelThe sparse autoencoder model should be compiled using adam optimization and binary cross-entropy lossAll layers should use areluactivation except for the last layer in the decoder, which should usesigmoid$ cat 1-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

autoencoder = __import__('1-sparse').autoencoder

SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
auto.fit(x_train, x_train, epochs=50,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./1-main.py
Epoch 1/50
235/235 [==============================] - 4s 15ms/step - loss: 0.2467 - val_loss: 0.1715
Epoch 2/50
235/235 [==============================] - 3s 14ms/step - loss: 0.1539 - val_loss: 0.1372
Epoch 3/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1316 - val_loss: 0.1242
Epoch 4/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1218 - val_loss: 0.1166
Epoch 5/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1157 - val_loss: 0.1122

...

Epoch 46/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0844 - val_loss: 0.0844
Epoch 47/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0843 - val_loss: 0.0840
Epoch 48/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0842 - val_loss: 0.0837
Epoch 49/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0841 - val_loss: 0.0837
Epoch 50/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0839 - val_loss: 0.0835
1/1 [==============================] - 0s 85ms/step
3.0174155
1/1 [==============================] - 0s 46ms/stepRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/autoencodersFile:1-sparse.pyHelp×Students who are done with "1. Sparse Autoencoder"Review your work×Correction of "1. Sparse Autoencoder"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

2. Convolutional AutoencodermandatoryWrite a functiondef autoencoder(input_dims, filters, latent_dims):that creates a convolutional autoencoder:input_dimsis a tuple of integers containing the dimensions of the model inputfiltersis a list containing the number of filters for each convolutional layer in the encoder, respectivelythe filters should be reversed for the decoderlatent_dimsis a tuple of integers containing the dimensions of the latent space representationEach convolution in the encoder should use a kernel size of(3, 3)with same padding andreluactivation, followed by max pooling of size(2, 2)Each convolution in the decoder, except for the last two, should use a filter size of(3, 3)with same padding andreluactivation, followed by upsampling of size(2, 2)The second to last convolution should instead use valid paddingThe last convolution should have the same number of filters as the number of channels ininput_dimswithsigmoidactivation and no upsamplingReturns:encoder, decoder, autoencoderis the encoder modeldecoderis the decoder modelautois the full autoencoder modelThe autoencoder model should be compiled using adam optimization and binary cross-entropy loss$ cat 2-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

autoencoder = __import__('2-convolutional').autoencoder

SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
print(x_test.shape)

encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)[:,:,:,0]

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i,:,:,0])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()
$ ./2-main.py
(60000, 28, 28, 1)
(10000, 28, 28, 1)
Epoch 1/50
235/235 [==============================] - 117s 457ms/step - loss: 0.2466 - val_loss: 0.1597
Epoch 2/50
235/235 [==============================] - 107s 457ms/step - loss: 0.1470 - val_loss: 0.1358
Epoch 3/50
235/235 [==============================] - 114s 485ms/step - loss: 0.1320 - val_loss: 0.1271
Epoch 4/50
235/235 [==============================] - 104s 442ms/step - loss: 0.1252 - val_loss: 0.1216
Epoch 5/50
235/235 [==============================] - 99s 421ms/step - loss: 0.1208 - val_loss: 0.1179

...

Epoch 46/50
235/235 [==============================] - 72s 307ms/step - loss: 0.0943 - val_loss: 0.0933
Epoch 47/50
235/235 [==============================] - 80s 339ms/step - loss: 0.0942 - val_loss: 0.0929
Epoch 48/50
235/235 [==============================] - 65s 279ms/step - loss: 0.0940 - val_loss: 0.0932
Epoch 49/50
235/235 [==============================] - 53s 225ms/step - loss: 0.0939 - val_loss: 0.0927
Epoch 50/50
235/235 [==============================] - 39s 165ms/step - loss: 0.0938 - val_loss: 0.0926
1/1 [==============================] - 0s 235ms/step
3.4141076
1/1 [==============================] - 0s 85ms/stepRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/autoencodersFile:2-convolutional.pyHelp×Students who are done with "2. Convolutional Autoencoder"Review your work×Correction of "2. Convolutional Autoencoder"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

3. Variational AutoencodermandatoryWrite a functiondef autoencoder(input_dims, hidden_layers, latent_dims):that creates a variational autoencoder:input_dimsis an integer containing the dimensions of the model inputhidden_layersis a list containing the number of nodes for each hidden layer in the encoder, respectivelythe hidden layers should be reversed for the decoderlatent_dimsis an integer containing the dimensions of the latent space representationReturns:encoder, decoder, autoencoderis the encoder model, which should output the latent representation, the mean, and the log variance, respectivelydecoderis the decoder modelautois the full autoencoder modelThe autoencoder model should be compiled using adam optimization and binary cross-entropy lossAll layers should use areluactivation except for the mean and log variance layers in the encoder, which should useNone,  and the last layer in the decoder, which should usesigmoid$ cat 3-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

autoencoder = __import__('3-variational').autoencoder

SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded, mu, log_sig = encoder.predict(x_test[:10])
print(mu)
print(np.exp(log_sig / 2))
reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()


l1 = np.linspace(-3, 3, 25)
l2 = np.linspace(-3, 3, 25)
L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

for i in range(25*25):
    ax = plt.subplot(25, 25, i + 1)
    ax.axis('off')
    plt.imshow(G[i].reshape((28, 28)))
plt.show()
$ ./3-main.py
Epoch 1/50
235/235 [==============================] - 5s 17ms/step - loss: 212.1680 - val_loss: 175.3891
Epoch 2/50
235/235 [==============================] - 4s 17ms/step - loss: 170.0067 - val_loss: 164.9127
Epoch 3/50
235/235 [==============================] - 4s 18ms/step - loss: 163.6800 - val_loss: 161.2009
Epoch 4/50
235/235 [==============================] - 5s 21ms/step - loss: 160.5563 - val_loss: 159.1755
Epoch 5/50
235/235 [==============================] - 5s 22ms/step - loss: 158.5609 - val_loss: 157.5874

...

Epoch 46/50
235/235 [==============================] - 4s 19ms/step - loss: 143.8559 - val_loss: 148.1236
Epoch 47/50
235/235 [==============================] - 4s 19ms/step - loss: 143.7759 - val_loss: 148.0166
Epoch 48/50
235/235 [==============================] - 4s 19ms/step - loss: 143.6073 - val_loss: 147.9645
Epoch 49/50
235/235 [==============================] - 5s 19ms/step - loss: 143.5385 - val_loss: 148.1294
Epoch 50/50
235/235 [==============================] - 5s 20ms/step - loss: 143.3937 - val_loss: 147.9027
1/1 [==============================] - 0s 124ms/step
[[-4.4424314e-04  3.7557125e-05]
 [-2.3759568e-04  3.6484184e-04]
 [ 3.6569734e-05 -7.3342602e-04]
 [-5.5730779e-04 -6.3699216e-04]
 [-5.8648770e-04  8.7332644e-04]
 [ 1.7586297e-04 -8.7016745e-04]
 [-5.4950645e-04  6.9131691e-04]
 [-5.1684811e-04  3.8412266e-04]
 [-2.7567835e-04  5.2892545e-04]
 [-5.0945382e-04  1.0410405e-03]]
[[0.9501978  3.0150387 ]
 [1.1207044  0.6665632 ]
 [0.19164634 1.5250858 ]
 [0.9454097  0.45243642]
 [1.5451298  1.2251403 ]
 [0.28436017 1.3658737 ]
 [0.97746277 1.234872  ]
 [1.7042938  1.5537287 ]
 [1.2055128  1.1579443 ]
 [0.9644342  1.6614302 ]]
1/1 [==============================] - 0s 46ms/step
5/5 [==============================] - 0s 3ms/stepRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/autoencodersFile:3-variational.pyHelp×Students who are done with "3. Variational Autoencoder"Review your work×Correction of "3. Variational Autoencoder"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Autoencoders.md`
