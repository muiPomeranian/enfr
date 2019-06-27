import matplotlib.pyplot as plt
import utils
from collections import Counter
'''
100K data
most common eng length: 27, with freq: 5105
most common fr length: 31, with freq: 4007

num of sentence where len<=50 for er/fr: 112832,71.41% / 102617,64.95%
num of sentence where len<=100 for er/fr: 112832, 881% / 102617, 90%

For 1M data info :
most common eng length: 94, with freq: 10572
most common fr length: 128, with freq: 9427

num of sentence where len<=100 for er/fr: 650249,32.8% / 540067,27.24%

all data are skewed to right
'''

train2_en_path, train2_fr_path = utils.get_local_path('en_fr100K/train_sm.en'), utils.get_local_path('en_fr100K/train_sm.fr')


def get_len_list(en_path, fr_path):
    eng_name, fr_name = en_path, fr_path

    with open(eng_name) as f:
        eng_contents = f.readlines()
    with open(fr_name) as f:
        fr_contents = f.readlines()

    res_en, res_fr = [], []

    count_en, count_fr = 0, 0

    total_len = 0
    for line_en,line_fr in zip(eng_contents, fr_contents):
        if len(line_en) <= 100:
            count_en+=1
        if len(line_fr) <= 100:
            count_fr+=1

        res_en.append(len(line_en))
        res_fr.append(len(line_fr))
        total_len += 1

    return res_en, res_fr, count_en, count_fr, total_len


en_lens, fr_lens, count_en, count_fr, total_len = get_len_list(train2_en_path, train2_fr_path)


plt.hist(en_lens, bins=500)
plt.ylabel('frequency')
plt.show()

counter_en, counter_fr = Counter(en_lens), Counter(fr_lens)


print('[INFO] most common eng length: {}, with freq: {}'.format(counter_en.most_common()[0][0],
                                                                counter_en.most_common()[0][1]))
print('[INFO] most common fr length: {}, with freq: {}\n'.format(counter_fr.most_common()[0][0],
                                                                 counter_fr.most_common()[0][1]))

print('[INFO] num of sentence where len<=100 for er/fr: {},{}% / {},{}%'.format(count_en,
                                                                               round(count_en / total_len * 100, 2),
                                                                               count_fr,
                                                                               round(count_fr / total_len * 100, 2)))
print('[INFO] total_train_len: {}'.format(total_len))


if __name__ == '__main__':
    pass


