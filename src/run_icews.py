import torch.nn.functional as F
from datetime import datetime
from torch import nn

from dataset_icews import Dataset_ICEWS
# from dataset_yago import Dataset_Yago
from dataset_wiki import Dataset_Wiki

# from model import *
from model import *
import numpy as np
from utils import *
import argparse
import logging
import torch
import time
import os
import utils
from collections import Counter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR

desc = 'Temporal RotatE for Temporal KGE'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14')
parser.add_argument('-model', help='model name', type=str, default='TS_RotatE')
parser.add_argument('-epochs', help='Number of epochs', type=int, default=250)
parser.add_argument('-bsize', help='Batch size', type=int, default=512)
parser.add_argument('-bs', '--block-size', help='block-size', type=int, default=2)

parser.add_argument('-init1', help='init1', type=float, default=5e-2)
parser.add_argument('-init2', help='init2', type=float, default=7e-2)
parser.add_argument('-init3', help='init3', type=float, default=5e-3)
parser.add_argument('-init4', help='init4', type=float, default=2e-1)

parser.add_argument('-lr', help='Learning rate', type=float, default=0.005)
parser.add_argument('-g', help='gamma', type=float, default=0.85)
parser.add_argument('-fc', '--fact_count', help='fact count', type=int, default=10)

parser.add_argument('-dim', help='Embedding dimension', type=int, default=256)
parser.add_argument('-sample_size', help='Negative sample size', type=int, default=500)
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4)
parser.add_argument('-reg1', help='regularization', type=float, default=0.0)
parser.add_argument('-reg2', help='regularization time', type=float, default=0.0)
parser.add_argument('-eval_step', help='evaluate model and validate each K epochs', type=int, default=10)

parser.add_argument('--local', help='with only local plausibility', action='store_true')
parser.add_argument('--block', help='with block matrix', action='store_true')
parser.add_argument('-st', '--structured', help='structured matrix', action='store_true')
parser.add_argument('-fp', '--fix-pattern', help='fixed structured matrix', action='store_true')
parser.add_argument('-it', help='with interval time', action='store_true')

parser.add_argument('-s', '--static', help='with only global plausibility', action='store_true')
parser.add_argument('-dig', '--diagonal', help='with diagonal 1', action='store_true')
parser.add_argument('-f', '--full', help='with full 2d transformation', action='store_true')

args = parser.parse_args()

def train():
    model = TASTER(args=args)
    model.cuda()

    logging.info('params num: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    loss_f = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = ExponentialLR(optimizer, gamma=args.g, last_epoch=-1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, start = 0.0, time.time()

        for facts, dates, mask_part, date_ids in dataset.iterate(batch=args.bsize, filter=True):
            optimizer.zero_grad()

            facts = facts.cuda()
            [heads, rels, tails] = [facts[ : , : , i] for i in range(3)]

            if not args.it:
                date_ids = date_ids.cuda()
                # print(date_ids.shape, heads.shape, rels.shape)
                scores = model(heads, rels, tails, date_ids, mask_part=mask_part)
            else:
                start_ids, end_ids = date_ids
                # print(start_ids.shape, heads.shape, rels.shape)
                start_ids, end_ids = start_ids.cuda(), end_ids.cuda()
                scores1 = model(heads, rels, tails, start_ids, mask_part=mask_part)
                scores2 = model(heads, rels + args.nrelation // 2, tails, end_ids, mask_part=mask_part)
                scores = 1 / 2 * (scores1 + scores2)
                
            l = torch.zeros(facts.shape[0]).long().cuda()
            loss = loss_f(scores, l) + model.reg_loss() + model.weight_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

        print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + args.model + "," + dataset.name + ")")
        print('cost: {}'.format(time.time() - start))
        
        if epoch % args.eval_step == 0:
            logging.info("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + args.model + "," + dataset.name + ")")
            logging.info("Evaluating the model...")
            stop, mrr = eval(model)
            if stop:
                logging.info("early stop, best MRR is {}".format(best))
                break
            
            scheduler.step()

def eval(model):
    model.eval()
    s = time.time()
    ranks_map = {}
    
    for f in ['test', 'valid', 'train']:
        ranks_map[f] = {'head' : [], 'tail' : []}
        for facts, dates, mask_part, date_ids in dataset.iterate(batch=8, name=f, eval=True):
            all_facts = facts.repeat(1, dataset.nentity, 1)
            rep_idx = 0 if mask_part == 'head' else 2
            all_facts[ : , : , rep_idx] = torch.arange(dataset.nentity).repeat(facts.shape[0], 1)
            all_facts = all_facts.cuda()
            [heads, rels, tails] = [all_facts[ : , : , i] for i in range(3)]

            facts, dates = facts.cuda(), dates.cuda()
        
            if not args.it:
                date_ids = date_ids.cuda()
                scores = model(heads, rels, tails, date_ids, mask_part=mask_part)
            else:
                start_ids, end_ids = date_ids
                start_ids, end_ids = start_ids.cuda(), end_ids.cuda()
                scores1 = model(heads, rels, tails, start_ids, mask_part=mask_part)
                scores2 = model(heads, rels + args.nrelation // 2, tails, end_ids, mask_part=mask_part)
                # scores2 = model(heads, rels, tails, end_ids, mask_part=mask_part)
                scores = 1 / 2 * (scores1 + scores2)

            for i, fact in enumerate(facts):
                fact, date = tuple(fact[0].cpu().numpy()), tuple(dates[i].cpu().numpy())

                if not args.it:
                    date_id = tuple(date_ids[i].cpu().numpy())
                    triplet = fact + date + date_id
                else:
                    start_id, end_id = tuple(start_ids[i].cpu().numpy()), tuple(end_ids[i].cpu().numpy())
                    triplet = fact + date + start_id + end_id

                ht_mask = dataset.get_mask_key(triplet, mask_part)

                tmp = scores[i][triplet[rep_idx]].item()
                idx = torch.from_numpy(dataset.mask_dict[mask_part][ht_mask])
                scores[i][idx] = -float('inf')
                scores[i][triplet[rep_idx]] = tmp

            ans = facts[ : , 0, 0] if mask_part == 'head' else facts[ : , 0, 2]
            
            ranks = torch.sum(scores > scores.gather(1, ans.view(-1, 1)), dim=1).cpu().numpy() + 1
            ranks_map[f][mask_part].append(ranks)
    
    metrics = {}
    for key in ranks_map:
        ranks = np.concatenate(ranks_map[key]['head'] + ranks_map[key]['tail'])
        metrics[key] = {}
        metrics[key]['MRR']    = np.mean(1 / ranks)
        metrics[key]['hit@1']  = np.mean(ranks == 1) 
        metrics[key]['hit@3']  = np.mean(ranks <= 3) 
        metrics[key]['hit@10'] = np.mean(ranks <= 10) 
        logging.info('{} metrics {}'.format(key, metrics[key]))
        
    global best, early_stop
    stop = False
    mrr = (metrics['test']['MRR'] + metrics['valid']['MRR'])
    # mrr = metrics['test']['MRR']
    if mrr > best:
        best, early_stop = mrr, 3
        logging.info('saving the best model...')
        torch.save(model, saved_path + '_best')
    else:
        logging.info('MRR {}'.format(mrr))
        early_stop -= 1
        if early_stop == 0:
            stop = True

    logging.info('eval time: {}'.format(time.time() - s))

    return stop, mrr

if 'icews' in args.dataset:
    dataset = Dataset_ICEWS(args) 
else:
    dataset = Dataset_Wiki(args) 

args.min_year = dataset.min_year
ckp_name = utils.get_ckp_name(args)

args.nentity  = dataset.nentity
if args.it:
    args.nrelation = dataset.nrelation * 2
    args.n_day = len(dataset.year2ids) * 2
else:
    args.n_day = len(dataset.date2id)
    args.nrelation = dataset.nrelation

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

model_name = args.model if not args.static else args.model[3:]

directory = os.path.join("../models/", model_name, args.dataset)
if not os.path.exists(directory):
    os.makedirs(directory)

log_dir = os.path.join("../logs/", model_name, args.dataset)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

date_str = datetime.today().strftime('%m-%d-%H')
logging.basicConfig(filename=os.path.join('{}/{}.log'.format(log_dir, date_str)), level=logging.DEBUG, format=LOG_FORMAT)
early_stop = 4

logging.info('-' * 5 + 'model:{}, args:{}'.format(model_name, args) + '-' * 5)

saved_path = os.path.join(directory, ckp_name + '-' + date_str)
best = 0

if __name__ == "__main__":
    train()
