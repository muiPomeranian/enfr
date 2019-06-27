# Machine Translation EN-FR

<p align="center">
  <img src="/assets/french-translation_img.png" width="497" height="370">
</p>

![ubuntu 16.04](https://img.shields.io/badge/ubuntu-16.04-blue.svg)
[![Python 3.7](https://img.shields.io/badge/python-3.6.8-blue.svg)](https://www.python.org/downloads/release/python-370/)
![PyTorch](https://img.shields.io/badge/pytorch-0.4.1-blue.svg)
![cuda 10.0](https://img.shields.io/badge/cuda-10-blue.svg)
![tqdm](https://img.shields.io/badge/tqdm-blue.svg)


### About 
This project is to use __[Transformer](https://arxiv.org/abs/1706.03762)__ for translation of en-fr custom dataset.


### Goal Of the Project
Following main goal will be achieved.
1. Build a meaningful translator of english to french.
2. Can be applied to any other type of language machine translation easily if data is provided.


### Dataset
The raw dataset used in this repo is data `data/fra.txt` which contains `100K` data.


### Environment
    conda env create -f env.yml
    conda activate transformer


### Data Split
    python prep/split_100K.py -data data/fra.txt
    # this splits data with 8:1:1 ratio
splited data will be stored at `en_fr100K`


### Data Preparation
    python preprocess.py -train_src en_fr100K/train_sm.en -train_tgt en_fr100K/train_sm.fr -valid_src en_fr100K/dev_sm.en -valid_tgt en_fr100K/dev_sm.fr -save_data enfr100K.pd100.pt
    
We use spaCy to make torch input for the model training and test. Make dictionary type of data from raw data. Again, spaCy is used to tokenize. Please refer to the Tokenizaion section.
After split the data, we create the vocab(word dictionary). make word2idx, idx2word dictionary to map after training.


### Train
    python train.py -data enfr100K.pd100.pt -log [exp_model] -save_model [exp_name] -epoch 90

add `-no_cuda` in case GPU is not available


### Split
We split data with the ratio of 8:2:2(train/dev/test). 
If evaluation will be tested by manual input by human, then we can use 9:1 train/dev for more resources. Also, there is a public
en-fr data of 2millions of pairs. This could be added to further work. By keeping original test set, we can compare the result.
Since this production meant to show general approach to NLP training, model follows the rule of thumbs on middle size of data case.

Use split2M.py to conduct training on 2Millions of en-fr pairs. Use split100K.py to conduct training on given 100K pairs of data. Data is 
sampled randomly. There is no repetition. Furthermore, we can use random sampling with weight when we have big data like 2M cases.


### Tokenization
We use spaCy API(open source library). 
For information extraction in tokenization steps, spaCy splits complex nested tokens. 
Original paper author in Transformer uses naive tokenization method such as lemmatization, Stemming(reducing a word to its root form). 
While referring to the pre-processing for natural language processing tasks, researchers encounter various scenarios where the same root produces the multiple different words. 
To normalize the word, stemming usually comes for the sake of uniformity of the words.

Whereas spaCy uses the syntactic tree structure to provide ‘better’ uniformity of the word for each sentences. 
spaCy integrated word vectors and syntactic tree structures to give better dependency parser and part-of-speech tagger. 
This ables transformer architecture to recognize dependency of words or tokens. 


### statistics
Unlike seq2seq model, Transformer architecture requires fixed size of input as of self-attention mechanism. 
Each input sentences have different length and unavoidable trimming/padding is executed by size of 100. We can see there is right skewed distribution of length for each en/fr data.
Fortunately, min and mode are similar for both en/fr. Also, we don't want to trim too much of sentences. Thus, we decide to cover 90% of data to possess most of the information from the input(size = 100).
If user decide to work with bigger data, this threshold can be reduced.  




### train
train.py train the model on Transformer architecture with parameter listed above. enfr100K.pd100.pt preprocessed dictionary formed data is used from preprocess.py.
train.py will made portfo.chkpt and will be used at evaluation.

Model will have positional encoding enables non sequential model to maintain the positional memory of inputs. Then encoder makes self-attention which will be combined with decoder which has target tokens sequentially.  

We used batch size 64 and trimmed/padded to max sequence length of 100. 
The distribution of word lengths for each sentence in French and English shows left tail skewed distribution and length of 100 was chosen. 
Most common length of English and French were 27, and 31. While paper used 50, this does not cover our data well. This will trimmed down 60% of sentences. 
Therefore, length of 100 to trim down was chosen to cover more than 70% of sentence fully to not trim too much sentence. 

We used 6 transformer layers for encoding - decoding layers, with 8 multi-head.

### evaluation
tranls_eval.py uses batch wise(size 64) test to evaluate the model after training. This uses check point `/enfr/enfr100K.pd100.pt` produced by train.py


### translate
bash file? test script <- user can test their own word


### train resume 



### metric
perplexity to observe the entropy information
accuracy - recall
bleu score with 1gram




### Training Result
Run the Tensor Board
```bash
$ tensorboard --logdir='./logs' --port=6006
```
 
<p align="center">
  <img src="/assets/tensorboard_img.png" width="550" height="500">
</p>

0.4133846407231509 <- bleu score of weight 1 0 0 0 


### Benefits of Transformer architecture over self attention in seq2seq model.
Self-attention method revolutionize the sequential model training such as RNN(GRU, LSTM, etc). 
As the model underwent each words or tokens, self attention allows model to look at other position within the same sentence and make positional representative architecture for a better encoder.

Attention in Sequential model maintains a hidden state with the incorporation of its representation of previous words/vectors 
it has processed with the current word. However, due to the curse(limitation) of long term dependency(even with the Gate function in cell), 
long-term relationship is not well propagated. Using tanh activation will make memory leakage for long sequence, and rely will produce unstable result. 
While seq2seq model lose partial information when model compresses the input information at encoding step, transformer never loses it since it uses attention while making encoding and decoding layers. 
Whereas, self-attention in the transformer allows model to understand the relevant relationship of each words into one the model is currently training by using all input sequences at once with positional encoding which gives positional(sequential) information to non-sequential model.

Similar to what attention does in seq2seq models, decoder has a helper attention layer which helps the decoder focus on relevant parts of the input sentences while iterating sequential input of paired target sources.



### limitation(in machine)
AWS used, due to cost restriction, time limitation, was able to train upto 90 epochs.


### limitation(in model)



질문: 

모델의 이해도 코드의 이해도 같은것도 요구해서 좀 자세하게 너어봤습니다 ... 더 좋게 정리할 수있는 방법이있다면
귀뜸해주시면 고쳐보겠습니다 .. ! 

smoothing 이 왜 쓰이나요 ? 
get_distribution.py <- 마지막부분 지우나 ? 어디다느나 ? 
weight decay ???????????

90epoch돈 텐소보드위한 train valid weight도 부탁드려요! 

텐소보드 하는거 실행어도 해야할꺼같습니다 .. !?
trnalsator 부분 부탁드립니다 감사해요 ..!!!!!

further inprovement, possible action: 
- BPE, sampling, resume from 90 points(100K) for 2M data?

