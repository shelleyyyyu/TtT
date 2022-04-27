import torch
from torch import nn
import torch.nn.functional as F
from bert import BERTLM
from data_loader import DataLoader
from crf_layer import DynamicCRF
import onnx
import tensorflow as tf
from data import CLS, SEP, PAD
from onnx_tf.backend import prepare
from torch.nn.utils.spectral_norm import SpectralNormStateDictHook, SpectralNormLoadStateDictPreHook

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters



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


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(
            name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break
    return module

def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            remove_spectral_norm(item)
        except Exception:
            pass

        for child in item.children():
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)

class myModel(nn.Module):
    def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, vocab,
                 loss_type='FC_FT_CRF'):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        # self.bert_model = spectral_norm(self.bert_model, name='weights')
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

    def forward(self, input_data):
        self.input_data = input_data
        if torch.cuda.is_available():
            self.input_data = self.input_data.cuda(self.device)
        seq_len, current_batch_size = self.input_data.size()
        sequence_representation = self.bert_model.work(self.input_data)[0]  # [seq_len, batch_size, embedding_size]
        if torch.cuda.is_available():
            sequence_representation = sequence_representation.cuda(self.device)  # [seq_len, batch_size, embedding_size]
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)  # [seq_len, batch_size, embedding_size]
        sequence_representation = sequence_representation.view(current_batch_size * seq_len, self.embedding_size)  # [seq_len * batch_size, embedding_size]
        sequence_emissions = self.fc(sequence_representation)  # [seq_len * batch_size, num_class]; num_class: 所有token in vocab
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size, self.num_class)  # [seq_len, batch_size, num_class]; num_class: 所有token in vocab
        sequence_emissions = sequence_emissions.transpose(0, 1)
        self.decode_scores, self.decode_result = self.CRF_layer.decode(sequence_emissions)
        return self.decode_result

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
    parser.add_argument('--onnx_path', type=str, default="model")
    return parser.parse_args()

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    #bert_model = spectral_norm(bert_model, name='weights')
    return bert_model

if __name__ == "__main__":
    args = parse_config()
    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(args.restore_ckpt_path)
    bert_model = init_empty_bert_model(bert_args, bert_vocab, args.gpu_id)

    id_label_dict, label_id_dict = {}, {}
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
        label_id_dict[label] = lid

    batch_size = args.batch_size
    number_class = len(id_label_dict)
    embedding_size = bert_args.embed_dim
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
                input_data = batchify(test_batch_text_list, bert_vocab)
                print(test_batch_tag_list)
                print(input_data)
                # Run the t2t model
                test_batch_result = model(input_data)

                # Result Post-processing
                result = []
                for test_result in test_batch_result:
                    test_tag_str = []
                    for tag in test_result.tolist():
                        char = id_label_dict[int(tag)]
                        if char != CLS and char != SEP and char != PAD:
                            test_tag_str.append(char)
                    result.append(''.join(test_tag_str))
    print('result: ', result)

    onnx_filename = args.onnx_path + '.onnx'
    input_names = ["input_1", "input_2"]
    output_names = ["output_1"]
    text_data =  list([['天', '津', '涌', '用', '定', '額', '發', '票', '<-MASK->', '<-SEP->']])
    input_data = batchify(text_data, bert_vocab)

    # Export onnx file
    print('Export ONNX file')
    torch.onnx.export(model,
                      input_data,
                      onnx_filename,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)

    onnx_model = onnx.load(onnx_filename)
    tf_exp = prepare(onnx_model)
    tf_exp.export_graph("t2t_ocr")  # export the model


    sess = tf.compat.v1.Session()
    metagraph = tf.compat.v1.saved_model.loader.load(sess, ['serve'], 't2t_ocr')
    sig = metagraph.signature_def['serving_default']
    input_dict = dict(sig.inputs)
    output_dict = dict(sig.outputs)
    print('input_dict', input_dict)
    print('output_dict', output_dict)
    input = input_dict['input_1'].name
    output = output_dict['output_1'].name
    out_list = sess.run(output, feed_dict={input: input_data})

    result = []
    for out in out_list:
        test_tag_str = []
        for tag in out.tolist():
            char = id_label_dict[int(tag)]
            if char != CLS and char != SEP and char != PAD:
                test_tag_str.append(char)
        result.append(''.join(test_tag_str))
    print('result: ', result)

