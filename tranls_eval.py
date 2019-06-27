''' Translate input text with trained model. '''

import torch
import torch.utils.data
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from dataset import paired_collate_fn, TranslationDataset
from transformer import Constants
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

import numpy as np
import math
import utils
import os

def cal_performance(pred, gold):
    """ Apply label smoothing if needed """

    gold = gold.contiguous().view(-1)
    # print("gold", gold.shape)
    # print("pred", pred.shape)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


def cal_loss(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    pred_token = pred.shape[0]
    gold_token = gold.shape[1]

    if pred_token > gold_token:
        pred = pred[:gold_token,:]
    elif pred_token < gold_token:
        gold = gold[:, :pred_token]

    gold = gold.contiguous().view(-1)

    loss = F.cross_entropy(pred,
                           gold,
                           ignore_index=Constants.PAD,
                           reduction='sum')
    return loss


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-log',
                        type=str,
                        default="./translate.log",
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-tgt', required=True,
                        help='Source sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-output', default='pred_ep90.txt.txt',
                        help="""Path to output the predictions 
                        (each line will be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_tgt_word_insts = read_instances_from_file(
        opt.tgt,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])
    test_tgt_insts = convert_instance_to_idx_seq(
        test_tgt_word_insts, preprocess_data['dict']['tgt'])

    assert opt.batch_size == 1, "batch_size required to be 1 in evaluation " \
                                "of perplexity"

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts,
            tgt_insts=test_tgt_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    translator = Translator(opt)

    # for accuracy
    n_total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    pred_output_dir = utils.get_local_path('test_output')

    if not os.path.exists(pred_output_dir):
        os.mkdir(pred_output_dir)

    with open(os.path.join(pred_output_dir,opt.output), 'w') as f:
        for batch in tqdm(test_loader,
                          mininterval=2,
                          desc='  - (Test)',
                          leave=False):
            src_seqs = batch[0]
            src_poss = batch[1]
            tgt_seqs = batch[2]

            all_hyp, all_scores, word_probs = translator.translate_batch(
                src_seqs, src_poss)

            word_probs = np.array(word_probs)
            word_probs = word_probs.squeeze(axis=1)
            word_probs = word_probs.squeeze(axis=1)
            word_probs = torch.from_numpy(word_probs)

            golds = tgt_seqs[:, 1:]
            # print("golds.shape", golds.shape)
            maxlen = golds.shape[1]
            # print("maxlen", maxlen)

            pred_seqs = []
            for hyp in all_hyp:
                # print("hyp", hyp)
                # print("original hyp shape: ", len(hyp))
                hyp = hyp[0]  # pill-off redundant
                # print("picked one hyp shape: ", len(hyp))
                hyplen = len(hyp)

                if hyplen < maxlen:
                    diff = maxlen - hyplen
                    hyp.extend([0] * diff)
                else:
                    hyp = hyp[:maxlen]
                pred_seqs.append(hyp)

            # print("np.array(pred_seqs).shape",
            #       np.array(pred_seqs).shape)
            # print("pred_seqs", pred_seqs)
            pred_seqs = torch.LongTensor(np.array(pred_seqs)).view(
                len(pred_seqs) * maxlen)
            # print("golds", golds)
            # print("golds.shape", golds.shape)
            n_correct = cal_performance(pred_seqs,
                                        golds)

            loss = cal_loss(word_probs,
                            golds)
            # print("loss", loss)

            non_pad_mask = golds.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            n_total_loss += loss

            # for pred.txt
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx]
                                          for idx in idx_seq])
                    f.write(pred_line + '\n')
            break

    accuracy = n_word_correct / n_word_total
    accu_log = 'accu: {} |'.format(accuracy)
    print('[Info] accuracy: ', accu_log)

    loss_per_word = n_total_loss / n_word_total
    ppl = math.exp(min(loss_per_word, 100))

    ppl_log = 'ppl: {} |'.format(ppl)
    print('[Info] ppl: ', ppl_log)

    with open(opt.log, 'a') as log_tf:
        log_tf.write(accu_log + '\n')
        log_tf.write(ppl_log + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()

