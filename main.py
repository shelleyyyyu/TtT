import torch
import random
from bert import BERTLM
from data import Vocab, CLS, SEP, MASK, PAD
from data_loader import DataLoader
import os
from funcs import *
import argparse
from da.augment_by_rule import DataAugmentationByRule
from TtTModel import TtTModel

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters

def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path)
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
        bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    if torch.cuda.is_available():
        bert_model = bert_model.cuda(device)
    if args.freeze == 1:
        for p in bert_model.parameters():
            p.requires_grad=False
    return bert_model, bert_vocab, bert_args

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    return bert_model

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--train_data',type=str)
    parser.add_argument('--dev_data',type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--label_data',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--freeze',type=int)
    parser.add_argument('--number_class', type = int)
    parser.add_argument('--number_epoch', type = int)
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
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--augment_percentage', type=float, default=1.0)
    parser.add_argument('--augment_method', default='by_rule', choices=['by_rule', 'by_auto'])
    parser.add_argument('--augment_type', default='contributed', choices=['contributed', 'average', 'insert', 'delete', 'substitution', 'paraphrase'])
    parser.add_argument('--augment_descending', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()

    # --- create model save path --- #
    directory = args.model_save_path
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 

    # TtTModel construction
    print ('Initializing model...')
    if args.start_from:
        bert_args, model_args, bert_vocab, model_parameters = extract_parameters(args.start_from)
        bert_model = init_empty_bert_model(bert_args, bert_vocab, args.gpu_id)
    else:
        bert_model, bert_vocab, bert_args = init_bert_model(args, args.gpu_id, args.bert_vocab)

    id_label_dict = {}
    label_id_dict = {}
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
        label_id_dict[label] = lid
    batch_size = int(args.batch_size * (1 + args.augment_percentage))
    number_class = len(id_label_dict)
    embedding_size = bert_args.embed_dim
    fine_tune = args.fine_tune
    loss_type = args.loss_type
    l2_lambda = args.l2_lambda
    model = TtTModel(bert_model, number_class, embedding_size, batch_size, args.dropout, args.gpu_id, bert_vocab, args, loss_type)
    if torch.cuda.is_available():
        model = model.cuda(args.gpu_id)
    print('Model construction finished.')

    print('Preparing data ...')
    # Data Preparation
    train_path, dev_path, test_path = args.train_data, args.dev_data, args.test_data
    train_max_len = args.training_max_len
    nerdata = DataLoader(train_path, dev_path, test_path, bert_vocab, train_max_len)
    print ('data is ready')

    if args.augment_type == 'insert':
        # 只增强 Insert
        weight = [0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif args.augment_type == 'substitution':
        # 只增强 Substitution
        weight = [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0]
    elif args.augment_type == 'paraphrase':
        # 只增强 Local Paraphrasing
        weight = [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    elif args.augment_type == 'delete':
        # 只增强 Delete
        weight = [0.0, 0.13, 0.13, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.13, 0.8]
    elif args.augment_type == 'contributed':
        # 按原数据集分布增强
        weight = [0.078, 0.027, 0.027, 0.102, 0.024, 0.182, 0.168, 0.204, 0.204, 0.006, 0.027, 0.168]
    elif args.augment_type == 'average':
        # 平均增强
        weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # data augmentation parameter setting
    dataAugmentationByRule = DataAugmentationByRule(weight, bert_vocab)

    # optimizer parameter setting
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # training parameter setting
    num_epochs = args.number_epoch
    training_data_num, dev_data_num, test_data_num = nerdata.train_num, nerdata.dev_num, nerdata.test_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = int(dev_data_num / batch_size) + 1#dev_data_num
    max_dev_f1 = -1
    max_dev_model_name = ''

    train_f1_list, train_precision_list, train_recall_list = [], [], []
    dev_f1_list, dev_precision_list, dev_recall_list, dev_acc_list, dev_ckpt_list = [], [], [], [], []

    prediction_max_len = args.prediction_max_len # 用来分块截取prediction的
    dev_eval_path = args.dev_eval_path
    final_eval_path = args.final_eval_path
    test_eval_path = args.test_eval_path

    acc_bs = 0.
    # Start to train epochs
    for epoch in range(num_epochs):
        loss_accumulated = 0.
        loss_crf_accumulated = 0.
        loss_ft_accumulated = 0.

        model.train()

        print ('-------------------------------------------')
        print ('%d epochs have run' % epoch)

        total_train_pred = list()
        total_train_true = list()
        batches_processed = 0
        best_acc = 0.0
        to_augment_text_list, to_augment_tag_list = [], []

        # Start to train steps in one epoch
        for train_step in range(train_step_num):
            batches_processed += 1
            acc_bs += 1
            optimizer.zero_grad()

            # Create next batch data
            train_batch_text_list, train_batch_tag_list, train_batch_out_list = nerdata.get_next_batch(args.batch_size, mode = 'train')
            # if is first batch, will not have augment data, random the train batch to pad the batch
            if len(to_augment_text_list) == 0 or len(to_augment_tag_list) == 0:
                index_list = random.sample([i for i in range(len(train_batch_text_list))], int(int(args.batch_size * (1+args.augment_percentage))-len(train_batch_text_list)))
                for index in index_list:
                    train_batch_text_list.append(train_batch_text_list[index])
                    train_batch_tag_list.append(train_batch_tag_list[index])
                    train_batch_out_list.append(train_batch_out_list[index])
            else:
                # if have previous augment batch, will not have augment data, random the train batch to pad the batch
                train_batch_text_list.extend(to_augment_text_list)
                train_batch_tag_list.extend(to_augment_tag_list)
                train_batch_out_list.extend(to_augment_out_list)
                to_augment_text_list, to_augment_tag_list = [], []
                # if have previous augment batch, will not have augment data, random the train batch to pad the batch
                if len(train_batch_text_list) != int(args.batch_size * (1+args.augment_percentage)):
                    index_list = random.sample([i for i in range(len(train_batch_text_list))], int(int(args.batch_size * (1+args.augment_percentage))-len(train_batch_text_list)))
                    for index in index_list:
                        train_batch_text_list.append(train_batch_text_list[index])
                        train_batch_tag_list.append(train_batch_tag_list[index])
                        train_batch_out_list.append(train_batch_out_list[index])

            # tag target matrix
            train_tag_matrix = process_batch_tag(train_batch_tag_list, nerdata.label_dict)
            # tag mask matrix
            train_mask_matrix = make_mask(train_batch_tag_list)
            # forward computation
            train_batch_result, train_loss, loss_crf, loss_ft, train_input_data, loss_list = model(train_batch_text_list, train_mask_matrix, train_tag_matrix, fine_tune, args.gamma)
            # Augment current training data for next round training
            loss_list = loss_list.view(-1, batch_size)
            sorted_loss_list, sorted_loss_index_list = torch.sort(loss_list[0], descending=args.augment_descending)
            to_augment_data_idxs = sorted_loss_index_list[:int(args.batch_size * args.augment_percentage)]
            to_augment_correct_data = [''.join([data for data in train_batch_out_list[idx] if data != CLS and data != MASK and data != SEP and data != PAD]) for idx in to_augment_data_idxs]
            augment_data = dataAugmentationByRule.augment(to_augment_correct_data)
            to_augment_text_list, to_augment_tag_list, to_augment_out_list  = nerdata.process_one_list(augment_data, to_augment_correct_data)
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            train_loss = train_loss + l2_lambda * l2_reg
            
            # update
            loss_accumulated += train_loss.item()
            loss_crf_accumulated += loss_crf
            loss_ft_accumulated += loss_ft
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            valid_train_batch_result = get_valid_predictions(train_batch_result, train_batch_tag_list, nerdata.label_dict)
            for i in range(batch_size):
                assert len(list(valid_train_batch_result[i])) == len(list(train_batch_tag_list[i]))
                total_train_pred.extend(list(valid_train_batch_result[i]))
                total_train_true.extend(list(train_batch_tag_list[i]))

            if acc_bs % args.print_every == 0:
                print ("gBatch %d, lBatch %d, loss %.5f, loss_crf %.5f, loss_ft %.5f" % \
                        (acc_bs, batches_processed, loss_accumulated / batches_processed,\
                         loss_crf_accumulated / batches_processed, loss_ft_accumulated / batches_processed))
        
            if acc_bs % args.save_every == 0:
                model.eval()
                gold_tag_list = []
                wrong_tag_list = []
                pred_tag_list = []
                with torch.no_grad():
                    with open(dev_eval_path, 'w', encoding = 'utf8') as o:
                        for dev_step in range(dev_step_num):
                            dev_batch_text_list, dev_batch_tag_list, dev_batch_out_list = nerdata.get_next_batch(batch_size = batch_size, mode = 'dev')
                            dev_tag_matrix = process_batch_tag(dev_batch_tag_list, nerdata.label_dict)
                            dev_mask_matrix = make_mask(dev_batch_tag_list)
                            dev_batch_result, _, _, _, dev_input_data, _ = model(dev_batch_text_list, dev_mask_matrix, dev_tag_matrix, fine_tune = False)

                            dev_text = ''
                            for token in dev_batch_text_list[0]:
                                dev_text += token + ' '
                            dev_text = dev_text.strip()

                            valid_dev_text_len = len(dev_batch_text_list[0])
                            dev_tag_str = ''
                            pred_tags = []
                            for tag in dev_batch_result[0][1:valid_dev_text_len + 1]:
                                dev_tag_str += id_label_dict[int(tag)] + ' '
                                pred_tags.append(int(tag))
                            dev_tag_str = dev_tag_str.strip()
                            out_line = dev_text + '\t' + dev_tag_str
                            o.writelines(out_line + '\n')
                            wrong_tag_list.append(dev_input_data[1:].t()[0].tolist())
                            gold_tag_list.append(dev_batch_tag_list[0])
                            pred_tag_list.append(pred_tags)
                    assert len(gold_tag_list) == len(pred_tag_list) 

                    right_true, right_false, wrong_true, wrong_false = 0, 0, 0, 0
                    all_right, all_wrong = 0, 0

                    for glist, plist, wlist in zip(gold_tag_list, pred_tag_list, wrong_tag_list):
                        for c, w, p in zip(glist, wlist, plist):
                            # Right
                            if w == c:
                                if p == c:
                                    #TN
                                    right_true += 1
                                else:
                                    #FP
                                    right_false += 1
                            else: # Wrong
                                if p == c:
                                    #TP
                                    wrong_true += 1
                                else:
                                    # FN
                                    wrong_false += 1

                    all_wrong = wrong_true + wrong_false
                    recall_wrong = wrong_true + wrong_false
                    correct_wrong_r = wrong_true / all_wrong
                    correct_wrong_p = wrong_true / (right_false + wrong_true)
                    correct_wrong_f1 = (2 * correct_wrong_r * correct_wrong_p) / (correct_wrong_r + correct_wrong_p + 1e-8)
                    correct_wrong_acc = (right_true + wrong_true) / ( right_true + wrong_true + right_false + wrong_false + 1e-8)
                    print('At epoch %d, official dev acc : %.4f, f1 : %.4f, precision : %.4f, recall : %.4f' % (epoch, correct_wrong_acc, correct_wrong_f1, correct_wrong_p, correct_wrong_r))

                    if correct_wrong_f1 > max_dev_f1:
                        ckpt_fname = directory + '/epoch_%d_dev_f1_%.3f' % (epoch, correct_wrong_f1)
                        max_dev_f1 = correct_wrong_f1
                        dev_acc_list.append(correct_wrong_acc)
                        dev_f1_list.append(correct_wrong_f1)
                        dev_precision_list.append(correct_wrong_p)
                        dev_recall_list.append(correct_wrong_r)
                        dev_ckpt_list.append(ckpt_fname)
                        torch.save({'args': args,
                                    'model': model.state_dict(),
                                    'bert_args': bert_args,
                                    'bert_vocab': model.bert_vocab
                                    }, ckpt_fname)
                model.train()



    max_dev_f1_idx = np.argmax(dev_f1_list)
    max_dev_f1 = dev_f1_list[max_dev_f1_idx]
    max_dev_precision = dev_precision_list[max_dev_f1_idx]
    max_dev_recall = dev_recall_list[max_dev_f1_idx]
    max_dev_acc = dev_acc_list[max_dev_f1_idx]
    max_dev_ckpt_fname = dev_ckpt_list[max_dev_f1_idx]

    print ('-----------------------------------------------------')
    print ('At this run, the maximum dev acc:%f, f1:%f, dev precision:%f, dev recall:%f; checkpoint filename:%s' % \
        (max_dev_acc, max_dev_f1, max_dev_precision, max_dev_recall, max_dev_ckpt_fname))
    print ('-----------------------------------------------------')

