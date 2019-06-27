# import nltk
import spacy
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

# refer this to better understanding on bleu_score, word2vec similarity
# https://stackoverflow.com/questions/32395880/calculate-bleu-score-in-python



pred_txt_path = '/Users/seungbochoi/downloads/pycharm_project_163/pred_ep90.txt'
gold_txt_path = '/Users/seungbochoi/downloads/pycharm_project_163/test_sm.fr'

nlp_fr = spacy.load('fr_core_news_sm')
def tokenize(nlp_model, str2tokenize):
    doc = nlp_model(str2tokenize)
    token_bag = []
    for token in doc:
        token_bag.append(str(token))
    return token_bag

def get_bleuscore(pred, gold, weights=(1,0,0,0)):
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
    print(score)

avg_score = total_score/num_sents

print(avg_score)








#
# with open(gold_txt_path) as f:
#     reference = f.readlines()
#
# sents = content.split('.')
#
# print('<s>' in content)



'    '
# sentence format: '<s> hi I am a boy . <\s>'
# cands = []
# for idx,sent in enumerate(sents):
#     if idx == 0:
#         sent = sent.split('<blank>')[-1]
#         sent = sent.split('\n')[0]
#         cands.append(sent)
#     else:
#         sent = sent.split('<blank>')[-1]
#         sent = sent.split('\n')[1]
#         cands.append(sent)
#
# print(len(cands))
#
# # for gold(test set)
#
# golds = []
# for line in reference:
#     golds.append(line.split('\n')[0])
#
# print(len(golds))




#
# '<\s>hi hi hello.<\s>bye bybybyby bye.'


# # check the original length of each file
# print(len(content))
# print(len(reference)) # 1D array, string component
#
#
# # to parse well for each sentence,
# passage = [] # make them in one big string
# for line in content:
#     for token in line.split():
#         passage.append(token)
#
# candidates = []
# temp = []
#
# nlp_fr = spacy.load('fr_core_news_sm')
#
#
# # candidates = [[spacy쓴 setence],[],[]..]
#
# for token in passage:
#     temp.append(token)
#     if temp[-1] == '.':
#         str_ = tokenize(nlp_fr, ' '.join(temp)) # output list of tokenized 1 sentence
#         candidates.append(' '.join(str_).lower())
#     if token == '.':
#         temp = []
#
# print('candidates after spacy')
# print(candidates[:4])
#
#
# print('len of candidates!!!!!: ',len(candidates))
# print('len of reference!!!!!: ', len(reference))
#
# # print(candidates[5], reference[5]) # this seems paired
#
#
#
# new_reference = []
# for r in reference:
#     refined_string = (' '.join(tokenize(nlp_fr, r))).rstrip().lower()  # remove new line, lower case
#     new_reference.append(refined_string)
#
# print('\n')
# print('below test new_reference format \n')
#
# print(candidates[4000])
# print(new_reference[4000])
#
# cands, refs = [],[]
# cands, refs = candidates[:4000], new_reference[:4000]
#
# # print(cands)
# print('\n\n\n\n\n')
# # print(refs)
#
# #
# candidates_bl,gold_bl = [],[]
# for sentence_cand, sentence_gold in zip(cands, refs): # later change here to candidates, new_reference
#     candidates_bl.append([sentence_cand.split(' ')[1:]]) # excluding <\s>
#     gold_bl.append(sentence_gold.split(' '))
#
# print('\n\n')
# print('test length: {}  {}'.format(len(candidates_bl), len(gold_bl)))
# print('\n\n\n')
# print(candidates_bl[-5:])
# print('\n\n')
# print(gold_bl[-5:])


# TODO:
# remove <\s> from candidates..

# https://www.nltk.org/_modules/nltk/translate/bleu_score.html
# use corpus_bleu ?
# candidates = [[['i','love','you]], [['i', 'hate', 'you']]]
# reference  = [ ['french french'] ,  ['french', 'french'] ]
#
# from nltk.translate.bleu_score import corpus_bleu
#
#
# score_1grm = corpus_bleu(refs, cands, weights=(1, 0, 0, 0))
# score_2grm = corpus_bleu(refs, cands, weights=(0, 1, 0, 0))
# score_3grm = corpus_bleu(refs, cands, weights=(0, 0, 1, 0))
# score_4grm = corpus_bleu(refs, cands, weights=(0, 0, 0, 1))
#
#
#
#
# print('4137 lines of BLEU SCORE 1gram: {}'.format(score_1grm))
# print('4137 lines of BLEU SCORE 2gram: {}'.format(score_2grm))
# print('4137 lines of BLEU SCORE 3gram: {}'.format(score_3grm))
# print('4137 lines of BLEU SCORE 4gram: {}'.format(score_4grm))



# reference = [['this', 'is', 'a', 'test']]
# candidate = ['this', 'is', 'a', 'test']
# # score = sentence_bleu(reference, candidate)
# # print(score)
# #
#

# gold = 'Je pense que nous pouvons nous aider l\', une l\'autre.'.lower()
# pred = 'je crois que nous pouvons aider .'
#






# score = sentence_bleu(gold_tokens, pred_tokens)
# print(score)









# answer_token = [tokenized_ans]
#
# tokenized = tokenize(nlp_fr, 'je crois que nous pouvons aider.')
# pred = [tokenized]
#
# print(answer)
# print(pred)
# score = sentence_bleu(answer_token ,pred)
#
# print(score)

if __name__ == '__main__':
    pass