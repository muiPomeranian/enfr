# import nltk
import spacy
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import argparse

# refer this to better understanding on bleu_score, word2vec similarity
# https://stackoverflow.com/questions/32395880/calculate-bleu-score-in-python


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pred', required=True)
    parser.add_argument('-data_gold', required=True)
    parser.add_argument('-n_gram', required=True)

    opt = parser.parse_args()

    pred_txt_path = opt.data_pred
    gold_txt_path = opt.data_gold

    if opt.n_gram == str(1):
        w = (1,0,0,0)
    elif opt.n_gram == str(2):
        w = (0,1,0,0)
    elif opt.n_gram == str(3):
        w = (0,0,1,0)
    elif opt.n_gram == str(4):
        w = (0,0,0,1)
    else:
        print('put n_gram less than equal to 4')

    nlp_fr = spacy.load('fr_core_news_sm')

    def tokenize(nlp_model, str2tokenize):
        doc = nlp_model(str2tokenize)
        token_bag = []
        for token in doc:
            token_bag.append(str(token))
        return token_bag

    def get_bleuscore(pred, gold, weights=w):
        gold_tokens = tokenize(nlp_fr, gold)
        pred_tokens = tokenize(nlp_fr, pred)
        score = sentence_bleu([gold_tokens], pred_tokens, weights=weights)

        return score


    # open file
    with open(pred_txt_path) as f:
        content = f.read()

    # open file
    with open(gold_txt_path) as f:
        golds = f.readlines()

    sents = content.split('<s>')[1:]

    cands = []
    for sent in sents:
        cand = sent.split('</s>')[0]
        cands.append(cand.rstrip())

    refs = []
    for gold in golds:
        refs.append(gold.rstrip())

    assert len(cands) == len(refs), 'length should be same'
    num_sents = len(cands)

    total_score = 0
    for pred, gold in tqdm(zip(cands, refs)):
        score = get_bleuscore(pred, gold)
        total_score += score

    avg_score = total_score/num_sents

    print(avg_score)


if __name__ == '__main__':
    main()