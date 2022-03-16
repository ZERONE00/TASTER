import os
import numpy as np
import pandas as pd
import time
import torch


def get_ckp_name(args):
    # 根据args获得保存checkpoint的名字
    ckp_name = 'dim={}_sample={}_dropout={}_lr={}_bsize={}_reg1={}_reg2={}_init1={}_init2={}_init3={}'\
    .format(args.dim, args.sample_size, args.dropout, args.lr, args.bsize, args.reg1, args.reg2, args.init1, args.init2, args.init3)
    return  ckp_name

def get_triples_map(dataset):
    triples_map = {'train':{'full':{}, 'head':{}, 'tail':{}, 'ht':{}, 'rel':{}, 'rel_nod':{}}, 
                    'test':{'full':{}, 'head':{}, 'tail':{}, 'ht':{}, 'rel':{}, 'rel_nod':{}}, 
                    'valid':{'full':{}, 'head':{}, 'tail':{}, 'ht':{}, 'rel':{}, 'rel_nod':{}}}
    for key in triples_map:
        data = dataset.data[key]
        for fact in data:
            # fact = fact.astype(np.int)
            h, r, t, d = fact[0], fact[1], fact[2], '/'.join([str(item) for item in fact[3:]])
            parts = {'head':tuple((h, r)), 'tail':tuple((t, r)), 'full':tuple((h, r, t)), 'ht':tuple((h, t)), 'rel':r, 'rel_nod':r}
            for p in parts:
                if parts[p] not in triples_map[key][p]:
                    triples_map[key][p][parts[p]] = set()

            triples_map[key]['full'][parts['full']].add(d)
            triples_map[key]['head'][parts['head']].add(t)
            triples_map[key]['tail'][parts['tail']].add(h)
            triples_map[key]['ht'][parts['ht']].add(r)
            triples_map[key]['rel'][parts['rel']].add(tuple((h, t, d)))
            triples_map[key]['rel_nod'][parts['rel_nod']].add(tuple((h, t)))
    
    return triples_map

# def build_res_df(model, dataset, args):
#     base_dir = '../results'
#     saved_dir = os.path.join(base_dir, args.model, args.dataset)
#     # if not os.path.exists(saved_dir):
#     #     os.mkdir(saved_dir)
#     os.makedirs(saved_dir, exist_ok=True)
#     triples_map = get_triples_map(dataset)
   
#     # for f in ['train', 'test', 'valid']:
#     for f in ['test', 'valid']:
#         rows, saved_path = ['head, relation, tail, date, head_rank, tail_rank'], os.path.join(saved_dir, '{}.csv'.format(f))
#         for r in dataset.rel2id:
#             if r not in triples_map[f]['rel']:
#                 continue

#             for item in triples_map[f]['rel'][r]:
#                 h, t, d = item
#                 key = [h, r, t]
#                 fact = np.array(key + [int(i) for i in d.split('/')])
#                 head_rank, _, _ = tester.testone(fact, sort=False)
#                 tail_rank, _, _ = tester.testone(fact, sort=False, head_or_tail='tail')
#                 row = '{}, {}, {}, {}, {}, {}'.format(h, r, t, d, head_rank, tail_rank)
#                 rows.append(row)

#         with open(saved_path, 'w') as f:
#             f.write('\n'.join(rows))


def get_res_df(model, dataset, args, evalf='test'):
    model.eval()
    base_dir = '../results'
    saved_dir = os.path.join(base_dir, args.model, args.dataset)
    saved_path = os.path.join(saved_dir, '{}_{}.csv'.format(evalf, get_ckp_name(args)))
    if os.path.exists(saved_path):
        df = pd.read_csv(saved_path)
        return df

    
    os.makedirs(saved_dir, exist_ok=True)
    triples_map = get_triples_map(dataset)

    s = time.time()
    data, ranks_ht = [], {'head' : [], 'tail' : []}

    for facts, dates, mask_part in dataset.iterate(batch=8, name=evalf, eval=True):
        all_facts = facts.repeat(1, dataset.nentity, 1)
        rep_idx = 0 if mask_part == 'head' else 2
        # print(all_facts.shape)
        all_facts[ : , : , rep_idx] = torch.arange(dataset.nentity).repeat(facts.shape[0], 1)

        all_facts, facts, dates = all_facts.cuda(), facts.cuda(), dates.cuda()
        [heads, rels, tails] = [all_facts[ : , : , i] for i in range(3)]
        [years, months, days] = [dates[ : , i] for i in range(3)]
        scores = model(heads, rels, tails, years, months, days, mask_part=mask_part)
        for i, fact in enumerate(facts):
            fact, date = tuple(fact[0].cpu().numpy()), tuple(dates[i].cpu().numpy())
            triplet = fact + date
            ht_mask = dataset.get_mask_key(triplet, mask_part=mask_part)
            for idx in dataset.mask_dict[mask_part][ht_mask]:
                if idx != triplet[rep_idx]:
                    scores[i][idx] = -float('inf') # filter evaluation

        ans = facts[ : , 0, 0] if mask_part == 'head' else facts[ : , 0, 2]
        
        ranks = torch.sum(scores > scores.gather(1, ans.view(-1, 1)), dim=1).cpu().numpy() + 1
        ranks_ht[mask_part].append(np.expand_dims(ranks, 1))

        if mask_part == 'head':
            facts = facts.squeeze(1).cpu().numpy()
            dates = np.expand_dims(np.array(['/'.join(date.astype('str')) for date in dates.cpu().numpy()]), 1)
            tmp = np.concatenate((facts, dates), axis=1)
            data.append(tmp)
    
    data = np.concatenate(data, axis=0)
    h_ranks, t_ranks = np.concatenate(ranks_ht['head'], axis=0), np.concatenate(ranks_ht['tail'], axis=0)
    print(data.shape, h_ranks.shape, t_ranks.shape)
    df = pd.DataFrame(data=np.concatenate((data, h_ranks, t_ranks), axis=1), 
    columns=['head', 'relation', 'tail', 'date', 'head_rank', 'tail_rank'])
    df.to_csv(saved_path, index=False)

    return df

def get_res_df2(model, dataset, args, evalf='test'):
    model.eval()
    base_dir = '../results'
    saved_dir = os.path.join(base_dir, args.model, args.dataset)
    saved_path = os.path.join(saved_dir, '{}_{}.csv'.format(evalf, get_ckp_name(args)))
    if os.path.exists(saved_path):
        df = pd.read_csv(saved_path)
        return df

    
    os.makedirs(saved_dir, exist_ok=True)
    triples_map = get_triples_map(dataset)

    s = time.time()
    data, ranks_ht = [], {'head' : [], 'tail' : []}

    for facts, dates, start_ids, end_ids, mask_part in dataset.iterate(batch=2, name='test', eval=True):
        all_facts = facts.repeat(1, dataset.nentity, 1)
        rep_idx = 0 if mask_part == 'head' else 2
        # print(all_facts.shape)
        all_facts[ : , : , rep_idx] = torch.arange(dataset.nentity).repeat(facts.shape[0], 1)

        all_facts, facts, dates = all_facts.cuda(), facts.cuda(), dates.cuda()
        [heads, rels, tails] = [all_facts[ : , : , i] for i in range(3)]
        start_ids, end_ids = start_ids.cuda(), end_ids.cuda()
        scores1 = model(heads, rels, tails, start_ids, mask_part=mask_part)
        scores2 = model(heads, rels, tails, end_ids, mask_part=mask_part)
        scores = 1 / 2 * (scores1 + scores2)

        for i, fact in enumerate(facts):
            fact, date = tuple(fact[0].cpu().numpy()), tuple(dates[i].cpu().numpy())
            triplet = fact + date
            ht_mask = dataset.get_mask_key(triplet, mask_part=mask_part)
            for idx in dataset.mask_dict[mask_part][ht_mask]:
                if idx != triplet[rep_idx]:
                    scores[i][idx] = -float('inf') # filter evaluation

        ans = facts[ : , 0, 0] if mask_part == 'head' else facts[ : , 0, 2]
        
        ranks = torch.sum(scores > scores.gather(1, ans.view(-1, 1)), dim=1).cpu().numpy() + 1
        ranks_ht[mask_part].append(np.expand_dims(ranks, 1))

        if mask_part == 'head':
            facts = facts.squeeze(1).cpu().numpy()
            dates = np.expand_dims(np.array(['/'.join(date.astype('str')) for date in dates.cpu().numpy()]), 1)
            tmp = np.concatenate((facts, dates), axis=1)
            data.append(tmp)
    
    data = np.concatenate(data, axis=0)
    h_ranks, t_ranks = np.concatenate(ranks_ht['head'], axis=0), np.concatenate(ranks_ht['tail'], axis=0)
    print(data.shape, h_ranks.shape, t_ranks.shape)
    df = pd.DataFrame(data=np.concatenate((data, h_ranks, t_ranks), axis=1), 
    columns=['head', 'relation', 'tail', 'date', 'head_rank', 'tail_rank'])
    df.to_csv(saved_path, index=False)

    return df