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
    python preprocess.py -train_src en_fr100K/train_sm.en -train_tgt en_fr100K/train_sm.fr -valid_src en_fr100K/dev_sm.en -valid_tgt en_fr100K/dev_sm.fr -save_data enfr100K.pd100.pt -max_len 100
    
This generates vocab set in `enfr100K.pd100.pt`
This uses __[spaCy](https://spacy.io/)__ for tokenization. 

Before use `preprocess.py`, run the following command

    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm


### Train
    python train.py -data enfr100K.pd100.pt -log [exp_name] -save_model [exp_name] -epoch 90

add `-no_cuda` in case GPU is not available. 

This will create `[exp_name].chkpt`. `accuracy` and `ppl` will be logged at `[exp_name].train.log` and `[exp_name].valid.log` or one can simply run  following command to use tensorboard

    tensorboard --logdir='./logs' --port=6006
   
<p align="center">
  <img src="/assets/tensorboard_img.png" width="550" height="500">
</p>


### Test
    python tranls_eval -model [exp_name].chkpt -src en_fr100K/test_sm.en -tgt en_fr100K/test_sm.fr -vocab en_fr100K/enfr100K.pd100.pt -beam_size 1
add `-no_cuda` in case GPU is not available.

one can use the pretrained __[checkpoint](https://drive.google.com/drive/u/0/folders/1afBjAbscZWOoMHvIIwA0jgx4UQEykQWw)__ to reproduce following result and 

    Accuracy: 0.43496310833506313,
     
    Average Loss per Word: 6.134334087371826,
    
    Perplexity: 461.4317184792621
to generate prediction output `test_output/pred_ep90.txt`.


### Evaluation
To get the bleu score of n-gram:
Use `[number]`<= 4 since few sentences in input data has short lengths.

One can run following command to see the test bleu score(0.4133846407231509, higher than Transformer paper, but different, smaller data set)

    python get_score.py -data_pred test_output/pred_ep90.txt -data_gold en_fr100K/test_sm.fr -n_gram [number]


### Translation
One can test with the trained model by following below instruction.
sh file uses pre-made translation_src.en file to translate.

    sh translate.sh

or if you want to make your own `[exp_name].en` or translation_src.en, make txt file under `enfr`.

    python translate.py -model portfo.chkpt -src [exp_name].en -vocab enfr100K.pd100.pt
    
add `-no_cuda` in case GPU is not available.
