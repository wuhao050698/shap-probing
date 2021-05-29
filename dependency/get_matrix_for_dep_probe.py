from transformers import BertModel, BertTokenizer
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import os
from .dep_parsing import match_tokenized_to_untokenized


def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_dep_matrix(args, model, tokenizer, dataset):
    ## 获取mask id
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()
    ## 选择embedding某一层，默认最后一层layers=12
    LAYER = int(args.layers)
    LAYER += 1  # embedding layer
    out = [[] for i in range(LAYER)]  # 当前层的输出
    for line in tqdm(dataset.tokens):
        sentence = [x.form for x in line][1:]

        tokenized_text = tokenizer.tokenize(' '.join(sentence))
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        ## tokenized_text = [CLS]+tokens+...+[SEP]
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(LAYER)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                # last_layer = model_outputs[0]
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                    np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((line, tokenized_text, init_matrix))

    for k, one_layer_out in enumerate(out):
        k_output = args.output_file.format(args.model_type, args.metric, args.data_split, str(k))
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()


if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--layers', default='12')

    # Data args
    parser.add_argument('--data_split', default='WSJ23')
    parser.add_argument('--dataset', default='constituency/data/WSJ/')
    parser.add_argument('--output_dir', default='./results/')

    parser.add_argument('--metric', default='dist', help='metrics for impact calculation, support [dist, cos] so far')
    parser.add_argument('--cuda', action='store_true', help='invoke to use gpu')

    parser.add_argument('--probe', default='constituency', help="dependency, constituency, discourse")

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.probe + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data_split = args.dataset.split('/')[-1].split('.')[0]
    args.output_file = args.output_dir + '/{}-{}-{}-{}.pkl'

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    print(args)
    from utils import ConllUDataset
    dataset = ConllUDataset(args.dataset)
    print(args)
    get_dep_matrix(args, model, tokenizer, dataset)

