
## Transformer Applications

### Description
0. DatasetmandatoryCreate the classDatasetthat loads and preps a dataset for machine translation:Class constructordef __init__(self):creates the instance attributes:data_train, which contains theted_hrlr_translate/pt_to_entf.data.Datasettrainsplit, loadedas_supervideddata_valid, which contains theted_hrlr_translate/pt_to_entf.data.Datasetvalidatesplit, loadedas_supervidedtokenizer_ptis the Portuguese tokenizer created from the training settokenizer_enis the English tokenizer created from the training setCreate the instance methoddef tokenize_dataset(self, data):that creates sub-word tokenizers for our dataset:datais atf.data.Datasetwhose examples are formatted as a tuple(pt, en)ptis thetf.Tensorcontaining the Portuguese sentenceenis thetf.Tensorcontaining the corresponding English sentenceUse a pre-trained tokenizer:use the pretrained modelneuralmind/bert-base-portuguese-casedfor theportuguesetextuse the pretrained modelbert-base-uncasedfor theenglishtextTrain the tokenizers with a maximum vocabulary size of2**13Returns:tokenizer_pt, tokenizer_entokenizer_ptis the Portuguese tokenizertokenizer_enis the English tokenizer$ cat 0-main.py
#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
$ ./0-main.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
tinham comido peixe com batatas fritas ?
did they eat fish and chips ?
<class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
<class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:0-dataset.pyHelp×Students who are done with "0. Dataset"Review your work×Correction of "0. Dataset"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/17pts

1. Encode TokensmandatoryUpdate the classDataset:Create the instance methoddef encode(self, pt, en):that encodes a translation into tokens:ptis thetf.Tensorcontaining the Portuguese sentenceenis thetf.Tensorcontaining the corresponding English sentenceThe tokenized sentences should include the start and end of sentence tokensThe start token should be indexed asvocab_sizeThe end token should be indexed asvocab_size + 1Returns:pt_tokens, en_tokenspt_tokensis anp.ndarraycontaining the Portuguese tokensen_tokensis anp.ndarray.containing the English tokens$ cat 1-main.py
#!/usr/bin/env python3

Dataset = __import__('1-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(data.encode(pt, en))
for pt, en in data.data_valid.take(1):
    print(data.encode(pt, en))
$ ./1-main.py
([8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41, 1499, 5524, 252, 4421, 15, 201, 84, 41, 300, 395, 693, 314, 17, 8193], [8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501, 898, 113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254, 16, 8193])
([8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239, 666, 33, 8193], [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193])
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:1-dataset.pyHelp×Students who are done with "1. Encode Tokens"0/7pts

2. TF EncodemandatoryUpdate the classDataset:Create the instance methoddef tf_encode(self, pt, en):that acts as atensorflowwrapper for theencodeinstance methodMake sure to set the shape of theptandenreturn tensorsUpdate the class constructordef __init__(self):update thedata_trainanddata_validateattributes by tokenizing the examples$ cat 2-main.py
#!/usr/bin/env python3

Dataset = __import__('2-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./2-main.py
tf.Tensor(
[8192   45  363  748  262   41 1427   15 7015  262   41 1499 5524  252
 4421   15  201   84   41  300  395  695  314   17 8193], shape=(25,), dtype=int64) tf.Tensor(
[8192  122  282  140 2164 2291 1587   14  140  391  501  898  113  240
 4451  129 2689   14  379  145  838 2216  508  254   16 8193], shape=(26,), dtype=int64)
tf.Tensor([8192 1274  209  380 4767  209 1937 6859   46  239  666   33 8193], shape=(13,), dtype=int64) tf.Tensor([8192  386  178 1836 2794  122 5953   31 8193], shape=(9,), dtype=int64)
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:2-dataset.pyHelp×Students who are done with "2. TF Encode"Review your work×Correction of "2. TF Encode"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

3. PipelinemandatoryUpdate the classDatasetto set up the data pipeline:Update the class constructordef __init__(self, batch_size, max_len):batch_sizeis the batch size for training/validationmax_lenis the maximum number of tokens allowed per example sentenceupdate thedata_trainattribute by performing the following actions:filter out all examples that have either sentence with more thanmax_lentokenscache the dataset to increase performanceshuffle the entire dataset using abuffer sizeequal to20000.split the dataset into padded batches of sizebatch_sizeprefetch the dataset usingtf.data.experimental.AUTOTUNEto increase performanceupdate thedata_validateattribute by performing the following actions:filter out all examples that have either sentence with more thanmax_lentokenssplit the dataset into padded batches of sizebatch_size$ cat 3-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.random.set_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./3-main.py
tf.Tensor(
[[8192 6633   29 ...    0    0    0]
 [8192  516 5468 ...    0    0    0]
 [8192  855 1038 ...    0    0    0]
 ...
 [8192 2500  121 ...    0    0    0]
 [8192   55  201 ...    0    0    0]
 [8192  363  936 ...    0    0    0]], shape=(32, 36), dtype=int64) tf.Tensor(
[[8192 5107   28 ...    0    0    0]
 [8192 5890 5486 ...    0    0    0]
 [8192  171  224 ...    0    0    0]
 ...
 [8192   46  315 ...    0    0    0]
 [8192  192  145 ...    0    0    0]
 [8192  282  136 ...    0    0    0]], shape=(32, 36), dtype=int64)
tf.Tensor(
[[8192 1274  209 ...    0    0    0]
 [8192  580  796 ...    0    0    0]
 [8192 3073  116 ...    0    0    0]
 ...
 [8192 2948   16 ...    0    0    0]
 [8192  333  981 ...    0    0    0]
 [8192  421 5548 ...    0    0    0]], shape=(32, 37), dtype=int64) tf.Tensor(
[[8192  386  178 ...    0    0    0]
 [8192   46  176 ...    0    0    0]
 [8192   46 4783 ...    0    0    0]
 ...
 [8192   46 1132 ...    0    0    0]
 [8192  135  145 ...    0    0    0]
 [8192  122  979 ...    0    0    0]], shape=(32, 38), dtype=int64)
$Note : The output may varyRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:3-dataset.pyHelp×Students who are done with "3. Pipeline"Review your work×Correction of "3. Pipeline"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

4. Create MasksmandatoryCreate the functiondef create_masks(inputs, target):that creates all masks for training/validation:inputsis a tf.Tensor of shape(batch_size, seq_len_in)that contains the input sentencetargetis a tf.Tensor of shape(batch_size, seq_len_out)that contains the target sentenceThis function should only usetensorflowoperations in order to properly function in the training stepReturns:encoder_mask,combined_mask,decoder_maskencoder_maskis thetf.Tensorpadding mask of shape(batch_size, 1, 1, seq_len_in)to be applied in the encodercombined_maskis thetf.Tensorof shape(batch_size, 1, seq_len_out, seq_len_out)used in the 1st attention block in the decoder to pad and mask future tokens in the input received by the decoder. It takes the maximum between a lookaheadmask and the decoder target padding mask.decoder_maskis thetf.Tensorpadding mask of shape(batch_size, 1, 1, seq_len_in)used in the 2nd attention block in the decoder.$ cat 4-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
import tensorflow as tf

tf.random.set_seed(0)
data = Dataset(32, 40)
for inputs, target in data.data_train.take(1):
    print(create_masks(inputs, target))
$ ./4-main.py
(<tf.Tensor: shape=(32, 1, 1, 36), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 36, 36), dtype=float32, numpy=
array([[[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 1, 36), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>)
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:4-create_masks.pyHelp×Students who are done with "4. Create Masks"Review your work×Correction of "4. Create Masks"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

5. TrainmandatoryTake your implementation of a transformer from ourprevious projectand save it to the file5-transformer.py. Note, you may need to make slight adjustments to this model to get it to functionally train.Write a the functiondef train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:Nthe number of blocks in the encoder and decoderdmthe dimensionality of the modelhthe number of headshiddenthe number of hidden units in the fully connected layersmax_lenthe maximum number of tokens per sequencebatch_sizethe batch size for trainingepochsthe number of epochs to train forYou should use the following imports:Dataset = __import__('3-dataset').Datasetcreate_masks = __import__('4-create_masks').create_masksTransformer = __import__('5-transformer').TransformerYour model should be trained with Adam optimization withbeta_1=0.9,beta_2=0.98,epsilon=1e-9The learning rate should be scheduled using the following equation withwarmup_steps=4000:Your model should use sparse categorical crossentropy loss, ignoring padded tokensYour model should print the following information about the training:Every 50 batches,  you should printEpoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}Every epoch, you should printEpoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}Returns the trained model$ cat 5-main.py
#!/usr/bin/env python3

import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.random.set_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
print(type(transformer))
$ ./5-main.py
Epoch 1, Batch 0: Loss 9.033271789550781, Accuracy 0.0
Epoch 1, Batch 50: Loss 8.9612398147583, Accuracy 0.0030440413393080235
Epoch 1, Batch 100: Loss 8.842384338378906, Accuracy 0.02672104351222515

...


Epoch 1, Batch 900: Loss 5.393615245819092, Accuracy 0.20997539162635803
Epoch 1, Batch 950: Loss 5.21303653717041, Accuracy 0.21961714327335358
Epoch 1, Batch 1000: Loss 5.040729999542236, Accuracy 0.2290351837873459
Epoch 1: Loss 5.018191337585449, Accuracy 0.2303551733493805
Epoch 2, Batch 0: Loss 1.6565858125686646, Accuracy 0.45769229531288147
Epoch 2, Batch 50: Loss 1.6106284856796265, Accuracy 0.4158884584903717
Epoch 2, Batch 100: Loss 1.5795631408691406, Accuracy 0.4239685535430908

...


Epoch 2, Batch 900: Loss 0.8802141547203064, Accuracy 0.4850142300128937
Epoch 2, Batch 950: Loss 0.8516387343406677, Accuracy 0.4866654574871063
Epoch 2, Batch 1000: Loss 0.8241745233535767, Accuracy 0.4883441627025604
Epoch 2: Loss 0.8208674788475037, Accuracy 0.4885943830013275
<class '5-transformer.Transformer'>
$Note: In this example, we only train for 2 epochs since the full training takes quite a long time. If you’d like to properly train your model, you’ll have to train for 20+ epochsRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/transformer_appsFile:5-transformer.py, 5-train.pyHelp×Students who are done with "5. Train"0/5pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Transformer_Applications.md`
