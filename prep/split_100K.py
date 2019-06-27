import random as rd
import utils

file = utils.get_local_path('data/fra.txt')

print(file)
with open(file) as f:
    contents = f.readlines()

english, french = [],[]

for line in contents:

    input_text, target_text = line.strip().split('\t')
    english.append(input_text)
    french.append(target_text)


def generate_random_idx_dev_test_train(total_len):
    dev_idx, test_idx, train_idx = [], [], []
    num_dev_test = int(total_len * 0.2) # 80% train

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


dev_idx, test_idx, train_idx = generate_random_idx_dev_test_train(len(english))

test_en = [english[idx] for idx in test_idx]
test_fr = [french[idx] for idx in test_idx]

dev_en = [english[idx] for idx in dev_idx]
dev_fr = [french[idx] for idx in dev_idx]

train_en = [english[idx] for idx in train_idx]
train_fr = [french[idx] for idx in train_idx]


# save it as txt file
def generate_txt(file_name, data):
    with open(file_name, 'a') as the_file:
        for line in data:
            the_file.write(line+'\n')


# generate txt files for 6 train/dev/test in en/fr
file_names = ['train_sm.en', 'train_sm.fr', 'dev_sm.en', 'dev_sm.fr', 'test_sm.en', 'test_sm.fr']
datas = [train_en, train_fr, dev_en, dev_fr, test_en, test_fr]


# use if we need to make txt file for train/dev/test
for file_name,data in zip(file_names, datas):
    generate_txt(file_name, data)

print(len(train_en), len(dev_en), len(test_en))

if __name__ == '__main__':
    pass

