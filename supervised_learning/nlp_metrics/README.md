
## Natural Language Processing - Evaluation Metrics

# What are the Applications of Natural Language Processing?

Natural Language Processing (NLP) is used in a wide range of applications where the goal is to enable machines to understand, interpret, and generate human language. Some key applications include:

1. **Machine Translation**: Translating text from one language to another (e.g., Google Translate).
2. **Sentiment Analysis**: Identifying the emotional tone (positive, negative, neutral) in text (e.g., social media monitoring, product reviews).
3. **Speech Recognition**: Converting spoken language into text (e.g., virtual assistants like Siri, Alexa).
4. **Chatbots and Virtual Assistants**: Automating customer service or conversational interfaces (e.g., chatbot for answering common queries).
5. **Text Summarization**: Automatically summarizing large texts into shorter, concise versions (e.g., summarizing news articles).
6. **Named Entity Recognition (NER)**: Identifying proper nouns, such as people, organizations, or locations in text (e.g., extracting information from documents).
7. **Part-of-Speech Tagging**: Assigning parts of speech (e.g., noun, verb, adjective) to each word in a sentence.
8. **Question Answering**: Automatically answering questions posed in natural language (e.g., systems like IBM Watson).
9. **Language Generation**: Creating human-like text based on input data (e.g., GPT models generating creative writing or responses).
10. **Text Classification**: Categorizing text into predefined categories (e.g., spam detection in emails).

# What is a BLEU Score?

The **BLEU (Bilingual Evaluation Understudy)** score is a metric used to evaluate the quality of machine-translated text by comparing it to one or more reference translations. It measures the overlap of n-grams (contiguous sequences of words) between the candidate and reference translations, focusing on precision.

- **Strengths**: It is straightforward, fast to compute, and correlates reasonably well with human judgments at the corpus level.
- **Weaknesses**: It does not consider grammatical correctness or meaning and can be overly sensitive to exact matches in word choice.

The BLEU score ranges from 0 to 1, with higher scores indicating closer matches to the reference translation.

# What is a ROUGE Score?

The **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** score is a set of metrics used to evaluate text summaries by comparing the overlap of n-grams and word sequences between a candidate summary and reference summaries. Unlike BLEU, which emphasizes precision, ROUGE focuses on recall, measuring how much of the reference summary is captured by the candidate summary.

There are several types of ROUGE scores:
- **ROUGE-N**: Measures the overlap of n-grams (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).
- **ROUGE-L**: Measures the longest common subsequence between the candidate and reference summaries.
- **ROUGE-S**: Measures skip-bigram overlap, where bigrams can skip intermediate words.

- **Strengths**: ROUGE is commonly used for evaluating automatic text summarization and tends to capture recall better than BLEU.
- **Weaknesses**: Like BLEU, ROUGE focuses on word-level overlap and does not account for grammatical correctness or meaning.

# What is Perplexity?

**Perplexity** is a metric used to evaluate language models. It measures how well a language model predicts a sample of text. Perplexity is the exponentiation of the cross-entropy between the predicted probability distribution and the actual distribution of the words in the test data.

- A **lower perplexity** indicates that the model is better at predicting the test data (i.e., it is less "perplexed" by the text).
- A **higher perplexity** indicates that the model is worse at predicting the next word in the sequence.

Perplexity is commonly used for evaluating models that generate text, such as language models, and is particularly useful in tasks like language modeling and machine translation.

# When Should You Use One Evaluation Metric Over Another?

Choosing the right evaluation metric depends on the task and the specific characteristics of the data or model being evaluated:

- **BLEU Score**: Best suited for machine translation or tasks where precision (exact word overlap) is important. Use BLEU when you need to evaluate how closely a candidate translation matches reference translations at the word or n-gram level.

- **ROUGE Score**: Ideal for text summarization tasks where recall is important. ROUGE is useful when you want to measure how much of the reference summary is captured by the candidate summary. It is also beneficial when you care about capturing key information, even if the wording is different.

- **Perplexity**: Best for evaluating language models where the goal is to measure how well the model predicts the likelihood of sequences of text. It is commonly used in generative tasks like text generation, machine translation, and next-word prediction.

In summary, you should:
- Use **BLEU** for tasks like machine translation that require precise word-level evaluation.
- Use **ROUGE** for tasks like summarization that need a broader assessment of how much content has been captured.
- Use **Perplexity** when evaluating the overall quality and predictive power of language models.



### Description
Question #0The BLEU score measures:A model’s accuracyA model’s precisionA model’s recallA model’s perplexity

Question #1The ROUGE score measures:A model’s accuracyA model’s precisionA model’s recallA model’s perplexity

Question #2Perplexity measures:The accuracy of a predictionThe branching factor of a predictionA prediction’s recallA prediction’s accuracy

Question #3The BLEU score was designed for:Sentiment AnalysisMachine TranslationQuestion-AnsweringDocument Summarization

Question #4What are the shortcomings of the BLEU score?It cannot judge grammatical accuracyIt cannot judge meaningIt does not work with languages that lack word boundariesA higher score is not necessarily indicative of a better translation

0. Unigram BLEU scoremandatoryWrite the functiondef uni_bleu(references, sentence):that calculates the unigram BLEU score for a sentence:referencesis a list of reference translationseach reference translation is a list of the words in the translationsentenceis a list containing the model proposed sentenceReturns: the unigram BLEU score$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
$ ./0-main.py
0.6549846024623855
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/nlp_metricsFile:0-uni_bleu.pyHelp×Students who are done with "0. Unigram BLEU score"Review your work×Correction of "0. Unigram BLEU score"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

1. N-gram BLEU scoremandatoryWrite the functiondef ngram_bleu(references, sentence, n):that calculates the n-gram BLEU score for a sentence:referencesis a list of reference translationseach reference translation is a list of the words in the translationsentenceis a list containing the model proposed sentencenis the size of the n-gram to use for evaluationReturns: the n-gram BLEU score$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/nlp_metricsFile:1-ngram_bleu.pyHelp×Students who are done with "1. N-gram BLEU score"Review your work×Correction of "1. N-gram BLEU score"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

2. Cumulative N-gram BLEU scoremandatoryWrite the functiondef cumulative_bleu(references, sentence, n):that calculates the cumulative n-gram BLEU score for a sentence:referencesis a list of reference translationseach reference translation is a list of the words in the translationsentenceis a list containing the model proposed sentencenis the size of the largest n-gram to use for evaluationAll n-gram scores should be weighted evenlyReturns: the cumulative n-gram BLEU score$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/nlp_metricsFile:2-cumulative_bleu.pyHelp×Students who are done with "2. Cumulative N-gram BLEU score"Review your work×Correction of "2. Cumulative N-gram BLEU score"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Natural_Language_Processing__Evaluation_Metrics.md`
