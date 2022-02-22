import sys
import torch
from torch import nn
import torch.nn.functional as F
import random
#### Load pretrained bert model
from bert import BERTLM
from google_bert import BasicTokenizer
from data import Vocab, CLS, SEP, MASK
import numpy as np
from data_loader import DataLoader
from crf_layer import DynamicCRF
import os
from funcs import *


def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters


def init_bert_model(args, device, bert_vocab):
    bert_ckpt = torch.load(args.bert_path)
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
                        bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    if torch.cuda.is_available():
        bert_model = bert_model.cuda(device)
    if args.freeze == 1:
        for p in bert_model.parameters():
            p.requires_grad = False
    return bert_model, bert_vocab, bert_args


def ListsToTensor(xs, vocab):
    batch_size = len(xs)
    lens = [len(x) + 2 for x in xs]
    mx_len = max(lens)
    ys = []
    for i, x in enumerate(xs):
        y = vocab.token2idx([CLS] + x) + ([vocab.padding_idx] * (mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data


def batchify(data, vocab):
    return ListsToTensor(data, vocab)


class myModel(nn.Module):
    def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, vocab,
                 loss_type='FC_FT_CRF'):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size, self.num_class)
        self.CRF_layer = DynamicCRF(num_class)
        self.loss_type = loss_type
        self.bert_vocab = vocab

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost)

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1))
        g = (1 - torch.clamp(p, min=0.01, max=0.99)) ** gamma
        # g = (1 - p) ** gamma
        cost = -g * torch.log(p + 1e-8)
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost), g.view(y.shape)

    def forward(self, text_data, in_mask_matrix, in_tag_matrix, fine_tune=False, gamma=None):
        current_batch_size = len(text_data)
        max_len = 0
        for instance in text_data:
            max_len = max(len(instance), max_len)
        seq_len = max_len + 1  # 1 for [CLS]]

        # in_mask_matrix.size() == [batch_size, seq_len]
        # in_tag_matrix.size() == [batch_size, seq_len]
        mask_matrix = torch.tensor(in_mask_matrix, dtype=torch.uint8).t_().contiguous()
        tag_matrix = torch.LongTensor(in_tag_matrix).t_().contiguous()  # size = [seq_len, batch_size]
        if torch.cuda.is_available():
            mask_matrix = mask_matrix.cuda(self.device)
            tag_matrix = tag_matrix.cuda(self.device)
        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, current_batch_size])

        # input text_data.size() = [batch_size, seq_len]
        data = batchify(text_data, self.vocab)  # data.size() == [seq_len, batch_size]
        input_data = data
        if torch.cuda.is_available():
            data = data.cuda(self.device)

        sequence_representation = self.bert_model.work(data)[0]  # [seq_len, batch_size, embedding_size]
        if torch.cuda.is_available():
            sequence_representation = sequence_representation.cuda(self.device)  # [seq_len, batch_size, embedding_size]
        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout,
                                            training=self.training)  # [seq_len, batch_size, embedding_size]
        sequence_representation = sequence_representation.view(current_batch_size * seq_len,
                                                               self.embedding_size)  # [seq_len * batch_size, embedding_size]
        sequence_emissions = self.fc(
            sequence_representation)  # [seq_len * batch_size, num_class]; num_class: 所有token in vocab
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size,
                                                     self.num_class)  # [seq_len, batch_size, num_class]; num_class: 所有token in vocab

        # bert finetune loss
        probs = torch.softmax(sequence_emissions, -1)
        if "FC" in self.loss_type:
            loss_ft_fc, g = self.fc_nll_loss(probs, tag_matrix, mask_matrix, gamma=gamma)
        else:
            loss_ft = self.nll_loss(probs, tag_matrix, mask_matrix)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1)
        mask_matrix = mask_matrix.transpose(0, 1)

        if "FC" in self.loss_type:
            # loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask = mask_matrix, reduction='token_mean', g=g.transpose(0, 1), gamma=gamma)
            loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix, reduction='token_mean',
                                          g=None, gamma=gamma)
        else:
            loss_crf = -self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix, reduction='token_mean')

        decode_result = self.CRF_layer.decode(sequence_emissions, mask=mask_matrix)
        self.decode_scores, self.decode_result = decode_result
        self.decode_result = self.decode_result.tolist()

        if self.loss_type == 'CRF':
            loss = loss_crf
            return self.decode_result, loss, loss_crf.item(), 0.0, input_data
        elif self.loss_type == 'FT_CRF':
            loss = loss_ft + loss_crf
            return self.decode_result, loss, loss_crf.item(), loss_ft.item(), input_data
        elif self.loss_type == 'FC_FT_CRF':
            loss = loss_ft_fc + loss_crf_fc
            return self.decode_result, loss, loss_crf_fc.item(), loss_ft_fc.item(), input_data
        elif self.loss_type == 'FC_CRF':
            loss = loss_crf_fc
            return self.decode_result, loss, loss_crf_fc.item(), 0.0, input_data
        else:
            print("error")
            return self.decode_result, 0, 0, 0, input_data


import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--label_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--freeze', type=int)
    parser.add_argument('--number_class', type=int)
    parser.add_argument('--number_epoch', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--prediction_max_len', type=int)
    parser.add_argument('--dev_eval_path', type=str)
    parser.add_argument('--test_eval_path', type=str)
    parser.add_argument('--final_eval_path', type=str)
    parser.add_argument('--l2_lambda', type=float)
    parser.add_argument('--training_max_len', type=int)
    parser.add_argument('--restore_ckpt_path', type=str)
    return parser.parse_args()

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    return bert_model

if __name__ == "__main__":
    args = parse_config()
    print('Initializing model...')

    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(args.restore_ckpt_path)
    bert_model = init_empty_bert_model(bert_args, bert_vocab, args.gpu_id)
    #bert_model, bert_vocab, bert_args = init_bert_model(args, args.gpu_id, args.bert_vocab)

    id_label_dict = {}
    label_id_dict = {}
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
        label_id_dict[label] = lid

    batch_size = args.batch_size
    number_class = len(id_label_dict)  # args.number_class
    embedding_size = bert_args.embed_dim
    fine_tune = args.fine_tune
    loss_type = args.loss_type
    l2_lambda = args.l2_lambda
    model = myModel(bert_model, number_class, embedding_size, batch_size, args.dropout, args.gpu_id, bert_vocab,
                    loss_type)
    model.load_state_dict(model_parameters)
    if torch.cuda.is_available():
        model = model.cuda(args.gpu_id)

    print('Model construction finished.')

    # Data Preparation
    train_path, test_path, test_path = args.train_data, args.test_data, args.test_data
    # label_path = args.label_data
    train_max_len = args.training_max_len
    nerdata = DataLoader(train_path, test_path, test_path, bert_vocab, train_max_len)
    print('data is ready')

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # --- training part ---#
    num_epochs = args.number_epoch
    training_data_num, test_data_num, test_data_num = nerdata.train_num, nerdata.test_num, nerdata.test_num
    test_step_num = test_data_num
    test_eval_path = args.test_eval_path

    model.eval()
    gold_tag_list = []
    wrong_tag_list = []
    pred_tag_list = []
    with torch.no_grad():
        with open(test_eval_path, 'w', encoding='utf8') as o:
            for test_step in range(test_data_num):
                test_batch_text_list, test_batch_tag_list = nerdata.get_next_batch(batch_size=1, mode='test')
                test_tag_matrix = process_batch_tag(test_batch_tag_list, nerdata.label_dict)
                test_mask_matrix = make_mask(test_batch_tag_list)
                test_batch_result, _, _, _, test_input_data = model(test_batch_text_list, test_mask_matrix, test_tag_matrix, fine_tune=False)

                test_text = ''
                for token in test_batch_text_list[0]:
                    test_text += token + ' '
                test_text = test_text.strip()

                valid_test_text_len = len(test_batch_text_list[0])
                test_tag_str = ''
                pred_tags = []
                for tag in test_batch_result[0][1:valid_test_text_len + 1]:
                    test_tag_str += id_label_dict[int(tag)] + ' '
                    pred_tags.append(int(tag))
                test_tag_str = test_tag_str.strip()
                out_line = test_text + '\t' + test_tag_str
                o.writelines(out_line + '\n')
                wrong_tag_list.append(test_input_data[1:].t()[0].tolist())
                gold_tag_list.append(test_batch_tag_list[0])
                pred_tag_list.append(pred_tags)
            # print(len(gold_tag_list))
            # print(len(pred_tag_list))
            # print(len(wrong_tag_list))
            # print(gold_tag_list[0])
            # print(pred_tag_list[0])
            # print(wrong_tag_list[0])
            # print([id_label_dict[w] for w in gold_tag_list[0]])
            # print([id_label_dict[w] for w in pred_tag_list[0]])
            # print([id_label_dict[w] for w in wrong_tag_list[0]])
            # exit()
            assert len(gold_tag_list) == len(pred_tag_list)
            right_true, right_false, wrong_true, wrong_false = 0, 0, 0, 0
            all_right, all_wrong = 0, 0

            for glist, plist, wlist in zip(gold_tag_list, pred_tag_list, wrong_tag_list):
                for c, w, p in zip(glist, wlist, plist):
                    if w == c:
                        if p == c:
                            right_true += 1
                        else:
                            right_false += 1
                    else:
                        if p == c:
                            wrong_true += 1
                        else:
                            wrong_false += 1

            all_wrong = wrong_true + wrong_false
            recall_wrong = wrong_true + wrong_false
            recall = wrong_true / all_wrong
            precision = wrong_true / (right_false + wrong_true)
            f1 = (2 * recall * precision) / (recall + precision + 1e-8)
            acc = (right_true + wrong_true) / (right_true + wrong_true + right_false + wrong_false + 1e-8)
            print('Official test acc : %.4f, f1 : %.4f, precision : %.4f, recall : %.4f' % (acc, f1, precision, recall))
