
## Natural Language Processing - Word Embeddings

### Description
Question #0Word2Vec uses:Character n-gramsSkip-gramsCBOWCo-occurrence matricesNegative sampling

Question #1GloVe uses:Character n-gramsSkip-gramsCBOWCo-occurrence matricesNegative sampling

Question #2FastText uses:Character n-gramsSkip-gramsCBOWCo-occurrence matricesNegative sampling

Question #3ELMo uses:Character n-gramsSkip-gramsCBOWCo-occurrence matricesNegative sampling

Question #4Which of the following can be used in conjunction with the others?Word2VecGloVeFastTextELMo

# Natural Language Processing (NLP)
Natural Language Processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans using natural language. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. Common applications of NLP include language translation, sentiment analysis, chatbots, and speech recognition.

# What is a Word Embedding?
A word embedding is a type of word representation that allows words to be represented as vectors in a continuous vector space. These vectors capture semantic meanings of words such that words with similar meanings have similar representations. Word embeddings are learned from large corpora of text and are used in various NLP tasks to improve the performance of machine learning models by providing meaningful numerical representations of words.

# What is Bag of Words (BoW)?
The Bag of Words (BoW) model is a simple and commonly used representation of text data in NLP. In BoW, a text (such as a sentence or document) is represented as an unordered collection of words, disregarding grammar and word order but keeping track of the number of occurrences of each word. The result is a sparse vector where each dimension corresponds to a unique word in the vocabulary, and the value is the count of the word in the text.

# What is TF-IDF?
TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The TF-IDF score increases with the number of times a word appears in a document but is offset by the frequency of the word in the entire corpus. This helps in identifying words that are important to a particular document but not too common across all documents.

- **Term Frequency (TF)**: The number of times a word appears in a document.
- **Inverse Document Frequency (IDF)**: A measure of how rare a word is across all documents in the corpus.

# What is CBOW?
Continuous Bag of Words (CBOW) is a neural network architecture used in the Word2Vec model. In CBOW, the goal is to predict a target word based on its surrounding context words. The model takes in the context words as input and tries to predict the target word, effectively learning word embeddings that capture the semantic relationships between words.

# What is a Skip-Gram?
Skip-Gram is another neural network architecture used in the Word2Vec model, but it works in the opposite way of CBOW. In Skip-Gram, the model takes a target word as input and tries to predict the context words surrounding it. This approach is particularly useful for capturing the semantics of rare words by learning which words are likely to appear around them.

# What is an n-gram?
An n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words, or base pairs according to the application. In NLP, n-grams typically refer to sequences of words. For example:
- **Unigram**: A single word (e.g., "word").
- **Bigram**: A sequence of two words (e.g., "word embedding").
- **Trigram**: A sequence of three words (e.g., "natural language processing").

N-grams are used in various NLP tasks to capture the context and structure of the text.

# What is Negative Sampling?
Negative Sampling is an optimization technique used in training models like Word2Vec. Instead of updating the weights for all words in the vocabulary during training, negative sampling only updates the weights for a small, randomly selected subset of "negative" samples (words that do not appear in the context). This significantly reduces the computational complexity and speeds up the training process.

# What is Word2Vec, GloVe, FastText, ELMo?

## Word2Vec
Word2Vec is a popular technique for learning word embeddings from large corpora of text. It uses neural network models, specifically the CBOW and Skip-Gram architectures, to learn vector representations of words. These embeddings capture the semantic meaning of words, allowing words with similar meanings to have similar vector representations.

## GloVe
GloVe (Global Vectors for Word Representation) is another method for generating word embeddings. Unlike Word2Vec, which relies on predicting context words, GloVe constructs word vectors by factorizing a word co-occurrence matrix. This approach captures both local and global statistics of the corpus, producing embeddings that reflect the overall structure of the language.

## FastText
FastText is an extension of Word2Vec developed by Facebook AI Research. It improves upon Word2Vec by representing each word as a bag of character n-grams, which helps in handling rare words and out-of-vocabulary words. This approach allows FastText to generate better word embeddings for languages with rich morphology or for words that were not seen during training.

## ELMo (Embeddings from Language Models)
ELMo is a deep contextualized word representation model that generates word embeddings by considering the entire context of a word in a sentence. Unlike traditional word embeddings, which assign a single vector to each word, ELMo generates different embeddings for a word depending on its context. ELMo is based on bidirectional LSTM networks and captures complex syntactic and semantic characteristics of language.



0. Bag Of WordsmandatoryWrite a functiondef bag_of_words(sentences, vocab=None):that creates a bag of words embedding matrix:sentencesis a list of sentences to analyzevocabis a list of the vocabulary words to use for the analysisIfNone, all words withinsentencesshould be usedReturns:embeddings, featuresembeddingsis anumpy.ndarrayof shape(s, f)containing the embeddingssis the number of sentences insentencesfis the number of features analyzedfeaturesis a list of the features used forembeddingsYou are not allowed to usegenismlibrary.$ cat 0-main.py
#!/usr/bin/env python3

bag_of_words = __import__('0-bag_of_words').bag_of_words

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
E, F = bag_of_words(sentences)
print(E)
print(F)
$ ./0-main.py
[[0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
 [1 0 0 0 2 0 0 1 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1]
 [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
['are' 'awesome' 'beautiful' 'cake' 'children' 'future' 'good'
 'grandchildren' 'holberton' 'is' 'learning' 'life' 'machine' 'nlp' 'no'
 'not' 'one' 'our' 'said' 'school' 'that' 'the' 'very' 'was']
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:0-bag_of_words.pyHelp×Students who are done with "0. Bag Of Words"Review your work×Correction of "0. Bag Of Words"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/5pts

1. TF-IDFmandatoryWrite a functiondef tf_idf(sentences, vocab=None):that creates a TF-IDF embedding:sentencesis a list of sentences to analyzevocabis a list of the vocabulary words to use for the analysisIfNone, all words withinsentencesshould be usedReturns:embeddings, featuresembeddingsis anumpy.ndarrayof shape(s, f)containing the embeddingssis the number of sentences insentencesfis the number of features analyzedfeaturesis a list of the features used forembeddings$ cat 1-main.py
#!/usr/bin/env python3

tf_idf = __import__('1-tf_idf').tf_idf

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
vocab = ["awesome", "learning", "children", "cake", "good", "none", "machine"]
E, F = tf_idf(sentences, vocab)
print(E)
print(F)
$ ./1-main.py
[[1.         0.         0.         0.         0.         0.
  0.        ]
 [0.5098139  0.60831315 0.         0.         0.         0.
  0.60831315]
 [0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.        ]]
['awesome' 'learning' 'children' 'cake' 'good' 'none' 'machine']
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:1-tf_idf.pyHelp×Students who are done with "1. TF-IDF"Review your work×Correction of "1. TF-IDF"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/5pts

2. Train Word2VecmandatoryWrite a functiondef word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):that creates , builds and trains agensimword2vecmodel:sentencesis a list of sentences to be trained onvector_sizeis the dimensionality of the embedding layermin_countis the minimum number of occurrences of a word for use in trainingwindowis the maximum distance between the current and predicted word within a sentencenegativeis the size of negative samplingcbowis a boolean to determine the training type;Trueis for CBOW;Falseis for Skip-gramepochsis the number of iterations to train overseedis the seed for the random number generatorworkersis the number of worker threads to train the modelReturns: the trained model$ cat 2-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(w2v.wv["computer"])
$ ./2-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-5.4084123e-03 -4.0024161e-04 -3.4630739e-03 -5.3525423e-03
  7.8537250e-03  6.0376106e-03 -7.2068786e-03  8.4706023e-03
  9.4194375e-03 -4.6773944e-03 -1.4714753e-03  7.7868701e-04
  3.1418847e-03 -1.1449445e-03 -7.0248209e-03  8.6203460e-03
  3.8405668e-03 -9.1897873e-03  6.2861182e-03  4.6401238e-03
 -6.3345446e-03  2.2874642e-03  3.3452510e-05 -9.4326939e-03
  8.5479887e-03  4.3843947e-03 -3.7956119e-03 -9.6801659e-03
 -8.1744418e-03  5.1590190e-03 -7.0132040e-03  2.5517345e-04
  7.9740928e-03  8.5820844e-03 -4.6414314e-03 -8.6783506e-03
 -1.0252714e-04  6.8263449e-03  2.4930835e-03 -8.6662006e-03
  3.0034208e-03 -3.1138016e-03 -5.4757069e-03 -1.3940263e-03
  7.4658301e-03  9.3212416e-03 -7.1789003e-03  1.2446367e-03
  5.2299835e-03 -4.8227082e-03 -4.5468416e-03 -5.1664864e-03
 -5.8076275e-03  7.7623655e-03 -5.6275711e-03 -5.4826117e-03
 -7.4911392e-03 -7.5089061e-03  5.5693723e-03 -4.2333854e-03
  6.0395217e-03  1.7224610e-03  7.1680485e-03  1.0818100e-03
  5.2833045e-03  6.1942148e-03 -8.7793246e-03  1.2095189e-03
 -9.0695143e-04 -4.2315759e-03 -9.5113518e-04 -1.7420733e-03
 -1.6348124e-04  6.3624191e-03  6.5098871e-03  2.5301289e-03
  4.2057564e-03  9.1815516e-03  2.7381873e-03 -2.6119126e-03
 -8.3582308e-03  1.0522294e-03 -5.3706346e-03  1.8784833e-03
 -9.4858548e-03  6.9658230e-03  8.8912249e-03 -7.0905304e-03
  6.3830256e-03 -1.8697941e-03 -9.1663310e-03  8.1991795e-03
  8.8182641e-03 -9.1386624e-03  1.8672824e-03  6.4541246e-03
  5.7970393e-03 -1.6923201e-03  7.1983398e-03  6.5960791e-03]
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:2-word2vec.pyHelp×Students who are done with "2. Train Word2Vec"Review your work×Correction of "2. Train Word2Vec"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

3. Extract Word2VecmandatoryWrite a functiondef gensim_to_keras(model):that converts agensimword2vecmodel to akerasEmbedding layer:modelis a trainedgensimword2vecmodelsReturns: the trainablekerasEmbeddingNote:  the weights can / will be further updated in Keras.$ cat 3-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model
gensim_to_keras = __import__('3-gensim_to_keras').gensim_to_keras

print(common_texts[:2])
w2v = word2vec_model(common_texts, min_count=1)
print(gensim_to_keras(w2v))
$ ./3-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
<keras.src.layers.core.embedding.Embedding object at 0x7f08126b8910>
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:3-gensim_to_keras.pyHelp×Students who are done with "3. Extract Word2Vec"Review your work×Correction of "3. Extract Word2Vec"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

4. FastTextmandatoryWrite a functiondef fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):that creates, builds and trains agenismfastTextmodel:sentencesis a list of sentences to be trained onvector_sizeis the dimensionality of the embedding layermin_countis the minimum number of occurrences of a word for use in trainingwindowis the maximum distance between the current and predicted word within a sentencenegativeis the size of negative samplingcbowis a boolean to determine the training type;Trueis for CBOW;Falseis for Skip-gramepochsis the number of iterations to train overseedis the seed for the random number generatorworkersis the number of worker threads to train the modelReturns: the trained model$ cat 4-main.py
#!/usr/bin/env python3

from gensim.test.utils import common_texts
fasttext_model = __import__('4-fasttext').fasttext_model

print(common_texts[:2])
ft = fasttext_model(common_texts, min_count=1)
print(ft.wv["computer"])
$ ./4-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-4.4518875e-04  1.9057443e-04  7.1344204e-04  1.5088863e-04
  7.3785416e-04  2.0828047e-03 -1.4264339e-03 -6.6978252e-04
 -3.9446630e-04  6.1643129e-04  3.7035978e-04 -1.7527672e-03
  2.0829479e-05  1.0929988e-03 -6.6954875e-04  7.9767447e-04
 -9.0742309e-04  1.9187949e-03 -6.9725298e-04  3.7622583e-04
 -5.0849823e-05  1.6160590e-04 -8.3575735e-04 -1.4309353e-03
  1.8365250e-04 -1.1365860e-03 -2.1796341e-03  3.3816829e-04
 -1.0266158e-03  1.9360909e-03  9.3765622e-05 -1.2577525e-03
  1.7052694e-04 -1.0470246e-03  9.1582153e-04 -1.1945128e-03
  1.2874184e-03 -3.1551000e-04 -1.1084992e-03  2.2345960e-04
  5.9021922e-04 -5.7232735e-04  1.6017178e-04 -1.0333696e-03
 -2.6842864e-04 -1.2489735e-03 -3.4248878e-05  2.0717620e-03
  1.0997808e-03  4.9419136e-04 -4.3252495e-04  7.6816598e-04
  3.0231036e-04  6.4548600e-04  2.5580439e-03 -1.2883682e-04
 -3.8391326e-04 -2.1800243e-04  6.5950496e-04 -2.8844117e-04
 -7.4177544e-04 -6.5318396e-04  1.4357771e-03  1.7945657e-03
  3.2790678e-03 -1.1300950e-03 -1.5527758e-04  4.3252096e-04
  2.0878548e-03  5.8326498e-04 -4.1506172e-04  1.1454885e-03
 -6.3745341e-05 -2.0422263e-03 -8.0344628e-04  2.0709851e-04
 -8.6796697e-04  7.6198514e-04 -3.0726698e-04  2.1699023e-04
 -1.4049197e-03 -1.9049532e-03 -1.1490833e-03 -3.2594264e-04
 -7.8721769e-04 -2.5946668e-03 -6.0526514e-04  9.3661918e-04
  5.8702513e-04  3.1111998e-04 -5.1438244e-04  4.9440534e-04
 -1.7251119e-03  5.4227427e-04 -7.4013631e-04 -4.8912101e-04
 -1.3722111e-03  2.1129930e-03  1.4438890e-03 -1.0972627e-03]
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:4-fasttext.pyHelp×Students who are done with "4. FastText"Review your work×Correction of "4. FastText"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

5. ELMomandatoryWhen training an ELMo embedding model, you are training:The internal weights of the BiLSTMThe character embedding layerThe weights applied to the hidden statesIn the text file5-elmo, write the letter answer, followed by a newline, that lists the correct statements:A. 1, 2, 3B. 1, 2C. 2, 3D. 1, 3E. 1F. 2G. 3H. None of the aboveRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/word_embeddingsFile:5-elmoHelp×Students who are done with "5. ELMo"Review your work×Correction of "5. ELMo"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/2pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Natural_Language_Processing__Word_Embeddings.md`
