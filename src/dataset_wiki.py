import os
import torch
import math
import numpy as np
from collections import Counter
from datetime import datetime
from datetime import timedelta

class Dataset_Wiki():
    def __init__(self, args):
        self.name = args.dataset
        self.base_dir = os.path.join('../datasets/', self.name)
        self.data = {'train' : None, 'valid' : None, 'test' : None}
        self.ent2id, self.rel2id, self.date2id = {}, {}, {}
        self.min_year = float('inf')
        self.sample_size = args.sample_size - 1
        self.start_year= -500
        self.end_year = 3000
        # self.fact_count = 100
        self.fact_count = 10

        self.process_time()
        self.build_data()
        
        self.nentity, self.nrelation = len(self.ent2id), len(self.rel2id)
        self.get_mask_dict()

    def build_data(self):
        for key in ["train", "valid", "test"]:
            path = os.path.join(self.base_dir, '{}.txt'.format(key))
            triplets = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    h, r, t, d1, d2 = line.split('\t')
                    d_id = self.get_id((d1, d2), self.date2id)
                    h = self.get_id(h, self.ent2id)
                    t = self.get_id(t, self.ent2id)
                    r = self.get_id(r, self.rel2id)
                    # [y, m, d] = [0, 0, 0]
                    start_year, end_year = self.get_year(d1), self.get_year(d2, 'end')
                    start_idx, end_idx = self.get_year_id(start_year), self.get_year_id(end_year, 'end')
                    # start_idx, end_idx = self.get_year_id(start_year), self.get_year_id(end_year)
                    
                    triplets.append((h, r, t, d_id, start_idx, end_idx))

            self.data[key] = triplets

    def to_date(self, year):
        return datetime.strptime('{}-01-01'.format(str(year+1000).zfill(4)), '%Y-%m-%d')
        datetime.strptime('92-01-01', '%y-%m-%d')

    def get_year(self, date, part='start'):
        try:
            start = datetime.strptime(date, '%Y-%m-%d')
            return start
        except Exception as e:
            if date[0]=='-':
                start = -int(date.split('-')[1])
            else:
                start = date.split('-')[0]
                if start =='####':
                    start = self.start_year if part == 'start' else self.end_year
                else:
                    start = int(start.replace('#', '0'))


            # return start
            return self.to_date(start)

    def get_year_id(self, year, part='start'):
        # if year == self.start_year: 
        if year ==self.to_date(self.start_year):
            idx = 0
            
        # elif year == self.end_year:
        elif year ==self.to_date(self.end_year):
            idx = len(self.year2ids) - 1

        else:
            for key, time_idx in sorted(self.year2ids.items(), key=lambda x : x[1]):
                if year >= key[0] and year <= key[1]:
                    idx = time_idx

        return idx if part == 'start' else idx + len(self.year2ids)
            

    def process_time(self):
        years, year_class = [], []
        for key in ["train", "valid", "test"]:
            path = os.path.join(self.base_dir, '{}.txt'.format(key))
            with open(path, 'r') as f:
                for line in f.readlines():
                    _, _, _, d1, d2 = line.split('\t')
                    d1, d2 = self.get_year(d1), self.get_year(d2, 'end')
                    for d in [d1, d2]:
                        # if d != self.start_year and d != self.end_year:
                        if d != self.to_date(self.start_year) and d != self.to_date(self.end_year):
                            years.append(d)
        
        years.sort()
        freq, self.year2ids, count = Counter(years), {}, 0

        for key in sorted(freq.keys()):
            count += freq[key]
            if count >= self.fact_count:
                year_class.append(key)
                count=0
        year_class[-1]=years[-1]

        pre_year = years[0]
        for i, yc in enumerate(year_class): 
            self.year2ids[(pre_year, yc)] = i
            # pre_year = yc + 1
            pre_year = yc + timedelta(days=1)

        self.year_class = year_class
        self.years = years

    def iterate(self, batch=512, name='train', filter=True, eval=False):
        # tensors = torch.tensor(self.data[name])
        # tensors = tensors[torch.randperm(tensors.shape[0])] # shuffle
        # n_batches = math.ceil(tensors.shape[0] / batch)
        tensors = torch.tensor(self.data[name])
        tensors1 = tensors[torch.randperm(tensors.shape[0])] # shuffle
        tensors2 = tensors[torch.randperm(tensors.shape[0])] # shuffle

        n_batches = math.ceil(tensors.shape[0] / batch)
        if eval and name == 'train':
            n_batches = math.ceil(5000 / batch)
    
        for i in range(n_batches):
            for mask_part in ['head', 'tail']:
                if not eval:
                    tensors = tensors1 if mask_part == 'tail' else tensors2
                elif name == 'train':
                    tensors = tensors[ : 5000]

                pos_batch = tensors[i * batch : min((i + 1) * batch, tensors.shape[0])]
                dates, start_ids, end_ids = pos_batch[ : , 3 : 4], pos_batch[ : , 4 : 5], pos_batch[ : , 5 : ]
                if eval:
                    facts = pos_batch[ : , : 3].unsqueeze(1)
                else:
                    facts = self.get_pos_neg_batch(pos_batch, mask_part=mask_part, filter=filter)
                    
                yield facts, dates, mask_part, (start_ids, end_ids)

    def get_pos_neg_batch(self, pos_batch, mask_part='head', filter=False):
        neg_samples = []
        rep_idx = 0 if mask_part == 'head' else 2

        neg_sample = np.random.randint(low=1, high=self.nentity, size = (pos_batch.shape[0], self.sample_size * 2))
        neg_sample = (neg_sample + pos_batch[ : , rep_idx].unsqueeze(1).numpy()) % self.nentity

        def sample(i):
            triplet = tuple(pos_batch[i].numpy())
            ht_mask = self.get_mask_key(triplet, mask_part=mask_part)
            mask = np.in1d(neg_sample[i], self.mask_dict[mask_part][ht_mask], assume_unique=True, invert=True)
            tmp_sample = neg_sample[i][mask]
            if tmp_sample.size < self.sample_size:
                # print('warning')
                tmp_sample = neg_sample[i]
            return tmp_sample[ : self.sample_size]

        if filter:
            # for i, triplet in enumerate(pos_batch):
            #     triplet = tuple(triplet.numpy())
            #     ht_mask = triplet[1 : ] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : ]
            #     mask = np.in1d(neg_sample[i], self.mask_dict[mask_part][ht_mask], assume_unique=True, invert=True)
            #     tmp_sample = neg_sample[i][mask]
            #     if tmp_sample.size < self.sample_size:
            #         # print('warning')
            #         tmp_sample = neg_sample[i]
            #     neg_samples.append(tmp_sample[ : self.sample_size])

            # neg_sample = np.stack(neg_samples, axis=0)
            # p = Pool(1)
            # neg_sample = p.map(sample, range(len(pos_batch)))
            for i in range(len(pos_batch)):
                neg_samples.append(sample(i))
            neg_sample = np.stack(neg_samples, axis=0)
        
        pos_facts = pos_batch[ : , : 3].unsqueeze(1)
        neg_facts = pos_facts.repeat(1, self.sample_size, 1)
        
        neg_facts[ : , : , rep_idx] = torch.LongTensor(neg_sample)[ : , : self.sample_size]

        return torch.cat((pos_facts, neg_facts), dim=1)
    
    def get_id(self, key, idmap):
        if key in idmap:
            return idmap[key]
        idmap[key] = len(idmap)
        return idmap[key]

    def get_mask_key(self, triplet, mask_part='head'):
        return triplet[1 : 4] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : 4]

    def get_mask_dict(self):
        self.mask_dict = {'head' : {}, 'tail' : {}}
        for key in self.data:
            for triplet in self.data[key]:
                head_mask, tail_mask = self.get_mask_key(triplet), self.get_mask_key(triplet, mask_part='tail')
                if head_mask not in self.mask_dict['head']:
                    self.mask_dict['head'][head_mask] = np.array([]).astype(int)
                tmp = self.mask_dict['head'][head_mask]
                self.mask_dict['head'][head_mask] = np.append(tmp, triplet[0])

                if tail_mask not in self.mask_dict['tail']:
                    self.mask_dict['tail'][tail_mask] = np.array([]).astype(int)
                tmp = self.mask_dict['tail'][tail_mask]
                self.mask_dict['tail'][tail_mask] = np.append(tmp, triplet[2])