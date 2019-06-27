from collections import defaultdict
import random as rd
import re
from unicodedata import normalize
import string
import utils
import os


#  generate dictionary {index : [english, french]}
def generate_enfr_dict(en_path, fr_path):
    english_name = en_path
    french_name = fr_path

    with open(english_name) as f:
        english_content = f.readlines()
    with open(french_name) as f:
        french_content = f.readlines()

    idx = 0
    en_fr_dict = defaultdict(list)  # {idx: [english, french]}

    for en_line, fr_line in zip(english_content, french_content):

        # sometimes one of them are missing, ignore to put it in
        if len(en_line.split()) == 0 or len(fr_line.split()) == 0:
            continue

        en_fr_dict[idx] = [en_line, fr_line]
        idx += 1

    return en_fr_dict


data_dir = utils.get_local_path('data')
en_path, fr_path = os.path.join(data_dir, 'europarl-v7.fr-en.en'), os.path.join(data_dir, 'europarl-v7.fr-en.fr')
en_fr_dict = generate_enfr_dict(en_path, fr_path)


# generate random index(1% out of entire data set)
# return list of idx
def generate_random_idx_dev_test_train(total_len):
    dev_idx, test_idx, train_idx = [], [], []
    num_dev_test = int(total_len * 0.01) # total number of dev+test set

    # does not allow repeated sampling in between dev/test set. Strictly independent
    total_idx = []

    set_ = set()
    for _ in range(num_dev_test):
        idx = rd.randint(0, total_len-1)
        while idx in set_:
            idx = rd.randint(0, total_len - 1)
        total_idx.append(idx)
        set_.add(idx)

    # divide total_idx half and half
    half_point = len(total_idx)//2
    dev_idx, test_idx = total_idx[:half_point], total_idx[half_point:]

    set_total_idx = set(total_idx)
    train_idx = [num for num in range(total_len) if num not in set_total_idx]

    return dev_idx, test_idx, train_idx


dev_idx, test_idx, train_idx = generate_random_idx_dev_test_train(len(en_fr_dict))

train_fr = [en_fr_dict[idx][1] for idx in train_idx]
train_en = [en_fr_dict[idx][0] for idx in train_idx]

test_fr = [en_fr_dict[idx][1] for idx in test_idx]
test_en = [en_fr_dict[idx][0] for idx in test_idx]

dev_fr = [en_fr_dict[idx][1] for idx in dev_idx]
dev_en = [en_fr_dict[idx][0] for idx in dev_idx]


# now convert these 6 sets to each seperated txt file
# file_name = txt file name
# data = which list will be used
def generate_txt(file_name, data):
    with open(file_name, 'a') as the_file:
        for line in data:
            the_file.write(line)


# generate txt files for 6 train/dev/test in en/fr
file_names = ['train2.en', 'train2.fr', 'dev2.en', 'dev2.fr', 'test2.en', 'test2.fr']
datas = [train_en, train_fr, dev_en, dev_fr, test_en, test_fr]


# stats. count the length of tokens (max, min, avg)
def count_max_len(file_dict):
    max_len, min_len = -float('inf'), float('inf')

    tot_len = 0
    count_1,count_2 = 0,0

    for _, sentence in file_dict.items():
        max_len = max(max_len, len(sentence[0].split()), len(sentence[1].split()))

        if len(sentence[0].split()) == 0:
            # print(sentence[0])
            count_1 += 1
        if len(sentence[1].split()) == 0:
            # print(sentence[1])
            count_2 += 1
        min_len = min(min_len, len(sentence[0].split()), len(sentence[1].split()))
        tot_len += (len(sentence[0].split()) + len(sentence[1].split()))

    print('missing pairs: ',count_1,count_2)
    avg_len = tot_len//(2*len(file_dict))

    return max_len, min_len, avg_len


max_, min_ ,avg_ = count_max_len(en_fr_dict)
print('max/min/avg length of sentences: {}/{}/{}'.format(max_, min_, avg_))


# use if we need to make txt file for train/dev/test
for file_name,data in zip(file_names, datas):
    generate_txt(file_name, data)



if __name__ == '__main__':
    pass