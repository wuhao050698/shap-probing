# nohup python run_shap.py >> 1.log 2>&1 &
# 0 108546
# 1 108751

import numpy as np
import itertools
from itertools import combinations
import torch
from copy import deepcopy
# from __future__ import absolute_import, division, print_function

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import glob
import logging
import os
import random
import itertools
import numpy as np
import torch
# import hedge_bert as hedge
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from copy import copy, deepcopy
# from pytorch_transformers import (WEIGHTS_NAME, BertConfig,BertForSequenceClassification, BertTokenizer)

# from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (convert_examples_to_features,output_modes, processors)
import time
logger = logging.getLogger(__name__)

torch.cuda.set_device(2)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    examples = processor.get_train_examples(args.data_dir)
    #print(examples)
    # examples = ['By comparison , it cost $ 103.7 million to build the NoMa infill Metro station , which opened in 2004 .']
    # label_list = ['pos','neg']
    output_mode = output_modes[task]
    
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                            cls_token_at_end=bool(args.model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(args.model_type in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    #for f in features:
    #  print(f.ori_token)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
	
class arg(object):
  def __init__(self):
    self.data_dir = './dataset/IMDB'
    self.model_type = 'bert'
    self.model_name_or_path = 'bert-base-uncased' 
    # self.model_name_or_path = 'gilf/english-yelp-sentiment'
    self.task_name = 'sst-2'
    self.output_dir = './output/IMDB'
    self.max_seq_length = 250
    self.start_pos = 0
    self.end_pos = 2000
    self.visualize = 1
    self.per_gpu_eval_batch_size = 1
    self.n_gpu = 1
    self.device = 'cuda'
args = arg()

def evaluate(args, model, tokenizer, eval_dataset, fileobject, start_pos=0, end_pos=100, vis=-1):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    count = start_pos
    start_time = time.time()
    results = []
    num =0
    for batch in itertools.islice(eval_dataloader, start_pos, end_pos):
        num+=1
        print(num)
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        count += 1
        fileobject.write(str(count))
        fileobject.write('\n')
        
        ori_text_idx = list(batch[0].cpu().numpy()[0])
        if 0 in ori_text_idx:
            ori_text_idx = [idx for idx in ori_text_idx if idx != 0]
        pad_start = len(ori_text_idx)
        print('len:',pad_start)
        with torch.no_grad():
            inputs = {'input_ids':      torch.unsqueeze(batch[0][0,:pad_start], 0),
                  'attention_mask': torch.unsqueeze(batch[1][0,:pad_start], 0),
                  'token_type_ids': torch.unsqueeze(batch[2][0,:pad_start], 0) if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                  'labels':         batch[3]}
            # print(inputs)
            outputs = model(**inputs)
            global tttest
            tttest = outputs[-1]
            # print(len(outputs))
            # print(outputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        # print(count,len(inputs['input_ids'][0]) - 2)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        for btxt in ori_text_idx:
            if (tokenizer.ids_to_tokens[btxt] != '[CLS]' and tokenizer.ids_to_tokens[btxt] != '[SEP]'):
                fileobject.write(tokenizer.ids_to_tokens[btxt])
                fileobject.write(' ')
        fileobject.write(' >> ')
        if batch[3].cpu().numpy()[0] == 0:
            fileobject.write('0')
            fileobject.write(' ||| ')
        else:
            fileobject.write('1')
            fileobject.write(' ||| ')

        shap = HEDGE(model, inputs, args, thre=100)
        res = shap.shapely_matrix(model,inputs)
        results.append(res)
    return results
	
class HEDGE:
    def __init__(self, model, inputs, args, max_level=-1, thre=0.3):
        score = model(**inputs)[1].detach().cpu().numpy()
        score_norm = (np.exp(score) / np.sum(np.exp(score), axis=1))
        self.pred_label = np.argmax(score_norm)
        self.max_level = max_level
        self.output = []
        self.fea_num = len(inputs['input_ids'][0]) - 2
        self.level = 0
        self.args = args
        self.thre = thre
        input = inputs['input_ids'][0]
        mask_input = torch.zeros(input.shape, dtype=torch.long)
        mask_attention = torch.zeros(input.shape, dtype=torch.long)
        mask_type = torch.zeros(input.shape, dtype=torch.long)
        temp = {'input_ids': torch.unsqueeze(mask_input, 0).to(args.device),
                'attention_mask': torch.unsqueeze(mask_attention, 0).to(args.device),
                'token_type_ids': torch.unsqueeze(mask_type, 0).to(args.device),
                'labels': inputs['labels']}
        # score = model(**temp)[1].detach().cpu().numpy()
        # score_norm = (np.exp(score) / np.sum(np.exp(score), axis=1))
        with torch.no_grad():
          ori_score = model(**temp)[-1][12].cpu().numpy()
          # ori_score = ori_score[0]
        self.ori_score = ori_score

    def set_contribution_func(self, model, fea_set, inputs, ori,tar):
      # input has just one sentence, input is a list
      args = self.args
      input = inputs['input_ids'][0]
      mask_input = torch.full(input.shape,tokenizer.mask_token_id)
      mask_input[0] = input[0]
      mask_input[-1] = input[-1]
      mask_attention = torch.zeros(input.shape, dtype=torch.long)
      mask_attention[0] = 1
      mask_attention[-1] = 1
      # mask_attention[ori+1] = 1
      # mask_input[ori+1] = input[ori+1]
      mask_type = torch.zeros(input.shape, dtype=torch.long)

      # mask the input with zero
      for fea_idx in fea_set:
          if type(fea_idx) == int:
              mask_input[fea_idx+1] = input[fea_idx+1] #+1 accounts for the CLS token at the begining
              mask_attention[fea_idx+1] = 1 #+1 accounts for the CLS token at the begining
          else:
              for idx in fea_idx:
                  mask_input[idx+1] = input[idx+1] #+1 accounts for the CLS token at the begining
                  mask_attention[idx+1] = 1 #+1 accounts for the CLS token at the begining
      temp = {'input_ids': torch.unsqueeze(mask_input, 0).to(args.device),
              'attention_mask': torch.unsqueeze(mask_attention, 0).to(args.device),
              'token_type_ids': torch.unsqueeze(mask_type, 0).to(args.device),
              'labels': inputs['labels']}
      # send the mask_input into model
      with torch.no_grad():
        score = model(**temp)[-1][12][:, ori+1, :].cpu().numpy()
        score = score[0]
      # score = score[12][:, word_pos, :] 
      # print(mask_attention)
      mask_input[tar+1] = input[tar+1]
      mask_attention[tar+1] = 1

      temp = {'input_ids': torch.unsqueeze(mask_input, 0).to(args.device),
              'attention_mask': torch.unsqueeze(mask_attention, 0).to(args.device),
              'token_type_ids': torch.unsqueeze(mask_type, 0).to(args.device),
              'labels': inputs['labels']}

      with torch.no_grad():
        score2 = model(**temp)[-1][12][:, ori+1, :].cpu().numpy()
        score2 = score2[0]
      # print(mask_attention)
      # print(score)
      # print(score2)
      return np.linalg.norm(score - score2)
      # return np.dot(score, score2) / (np.linalg.norm(score) * np.linalg.norm(score2))

    def shapely_matrix(self,model,inputs):
      fea_num = self.fea_num
      if fea_num == 0:
        return -1
      fea_set = list(range(fea_num))
      scores = [[] for i in range(len(fea_set))]
      for i in fea_set:
        for j in fea_set:
          if i==j:
            scores[i].append(0)
            continue
          # print(i,j)
          new_fea_set = [ele for x, ele in enumerate(fea_set) if x!=j and x!=i]
          # print(new_fea_set)
          scores[i].append(self.shapely_value(model,inputs,new_fea_set,i,j))
      return scores

    def get_shapley_interaction_weight(self, d, s):
      return np.math.factorial(s) * np.math.factorial(d - s - 1) / np.math.factorial(d)

    def shapely_value(self,model,inputs,feature_set,ori,tar):
      fea_num = len(feature_set)
      #dict_subset = {r: list(combinations(feature_set, r)) for r in range(fea_num+1)}
      score = 0.0
      for i in range(fea_num+1):
        # if i<(fea_num+1)*0.9:
        #   continue
        if i<fea_num-1:
          continue
        weight = self.get_shapley_interaction_weight(fea_num+1, i)
        if i==0:
          continue
        i_subsets = list(combinations(feature_set,i))
        for subsets in i_subsets:
          contri = self.set_contribution_func(model,list(subsets),inputs,ori,tar)
          score += contri*weight

      # print(score)
      return score

def print_out(num):
	print(num)

from transformers import BertModel, BertTokenizer,BertForSequenceClassification,RobertaModel,RobertaTokenizer,RobertaForSequenceClassification
# MODEL_CLASSES = {
#         'bert': (BertModel, BertTokenizer, 'gilf/english-yelp-sentiment'),
#         'roberta': (RobertaModel, RobertaTokenizer,'roberta-base')
#     }
MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES['bert']
# model = RobertaForSequenceClassification.from_pretrained(pretrained_weights, output_hidden_states=True)
model = BertForSequenceClassification.from_pretrained(pretrained_weights, output_hidden_states=True,cache_dir='cache_dir')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True,cache_dir='cache_dir')
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
# tokenizer = AutoTokenizer.from_pretrained("barissayil/bert-sentiment-analysis-sst",use_fast=False, do_lower_case=True)

# model = AutoModelForSequenceClassification.from_pretrained("barissayil/bert-sentiment-analysis-sst",output_hidden_states=True)

model.to('cuda')


from utils import ConllUDataset
dataset = ConllUDataset('PUD.conllu')

from utils_glue_txt import (convert_examples_to_features,output_modes, processors)
# from utils_glue import (convert_examples_to_features,output_modes, processors)
test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='test')

res = []
start_pos = args.start_pos
end_pos = args.end_pos
cnt = 0
logger.info(1111)


with open('test.txt', 'w') as f:
  print_out(cnt)
  logger.info(cnt)
  cnt +=1
  res = evaluate(args, model, tokenizer, test_dataset, f, start_pos, end_pos, args.visualize)
  
res2 = res
for line in res2:
  for i in line:
    i.insert(0,0)
    i.append(0)
  line.insert(0,[0 for i in range(len(line[0]))])
  line.append([0 for i in range(len(line[0]))])
  
num = 0
sum = 0
sens = []
in_result = []
for line in tqdm(dataset.tokens):
  sentence = [x.form for x in line][1:]
  # print(sentence)
  leng = len(tokenizer.tokenize(' '.join(sentence)))
  if num==0:
    num+=1
    continue
  if num>=100:
    break
  print_out(num)
  token_text = tokenizer.tokenize(' '.join(sentence))
  token_text.insert(0, '[CLS]')
  token_text.append('[SEP]')
  # print(len(res[num-1]))
  # print(len(line))
  # print(len(res[num-1]))
  in_result.append((line,token_text,res2[num-1]))
  # print(line)
  sen = ' '.join(sentence)
  sens.append(sen)
  sum += leng
  num += 1


from dependency import _evaluation as dep_eval
from dependency.dep_parsing import decoding as dep_parsing
class arg2(object):
  def __init__(self):
    self.probe = 'discourse'
    self.decoder = 'eisner'
    self.subword = 'avg'
    self.root = 'gold'
args = arg2()
trees, results, deprels = dep_parsing(in_result[0:99],args)
dep_eval(trees, results)