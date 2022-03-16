import os
import torch
import math
import numpy as np
from collections import Counter
import datetime

class Dataset_ICEWS():
    def __init__(self, args):
        self.name = args.dataset
        self.base_dir = os.path.join('../datasets/', self.name)
        self.data = {'train' : None, 'valid' : None, 'test' : None}
        self.ent2id, self.rel2id, self.date2id = {}, {}, {}
        self.min_year = float('inf')
        self.sample_size = args.sample_size - 1
        # self.fact_count = 200 if '14' in self.name else 10
        self.fact_count = args.fact_count

        self.process_time()
        base_dir = os.path.join('../datasets/', self.name)
        self.build_dict(base_dir)
        self.build_data()
        
        self.nentity, self.nrelation = len(self.ent2id) + 1, len(self.rel2id)
        print('nentity : {}, ntimes : {}'.format(self.nentity, len(self.date_class)))
        self.get_mask_dict()

    def todate(self, dstr):
        return datetime.datetime.strptime(dstr, "%Y-%m-%d").date()

    def build_dict(self, base_dir):
        path = os.path.join(base_dir, '{}.txt'.format('train'))
        with open(path, 'r') as f:
            for line in f.readlines():
                h, r, t, d = line.split('\t')
                self.get_id(h, self.ent2id)
                self.get_id(t, self.ent2id)

    def todateid(self, date):
        left, right = 0, len(self.date_class) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.date_class[mid] < date:
                left = mid + 1
            else:
                right = mid - 1
        
        return left

    def build_data(self):
        for key in ["train", "valid", "test"]:
        # for key in ["train"]:
            path = os.path.join(self.base_dir, '{}.txt'.format(key))
            triplets = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    h, r, t, d = line.split('\t')
                    h, t = h.strip(), t.strip()
                    date = self.todate(d.strip())
                    [y, m, d] = [int(item) for item in d.split('-')]
                    d_id = self.todateid(date)
                    # h = self.get_id(h, self.ent2id)
                    # t = self.get_id(t, self.ent2id)
                    # r = self.get_id(r, self.rel2id)
                    h = self.ent2id.get(h, len(self.ent2id))
                    t = self.ent2id.get(t, len(self.ent2id))
                    r = self.get_id(r, self.rel2id)
                    
                    triplets.append((h, r, t, y, m, d, d_id))

            self.data[key] = triplets

    def process_time(self):
        dates, self.date_class = [], []
        for key in ["train", "valid", "test"]:
            path = os.path.join(self.base_dir, '{}.txt'.format(key))
            with open(path, 'r') as f:
                for line in f.readlines():
                    _, _, _, d = line.split('\t')
                    d = self.todate(d.strip())
                    dates.append(d)
        
        dates.sort()
        self.freq, count = Counter(dates), 0

        for key in sorted(self.freq.keys()):
            count += self.freq[key]
            if count >= self.fact_count:
                self.date_class.append(key)
                count=0

        if key != self.date_class[-1]:
            self.date_class.append(key)

        for i, yc in enumerate(self.date_class): 
            self.date2id[yc] = i

    def iterate(self, batch=512, name='train', filter=True, eval=False):
        tensors = torch.tensor(self.data[name])
        tensors1 = tensors[torch.randperm(tensors.shape[0])] # shuffle
        tensors2 = tensors[torch.randperm(tensors.shape[0])] # shuffle

        n_batches = math.ceil(tensors.shape[0] / batch)
        if eval and name == 'train':
            n_batches = math.ceil(5000 / batch)

        fs = ['head', 'tail'] if eval else ['head', 'tail']
        for i in range(n_batches):
            for mask_part in fs:
                if not eval:
                    tensors = tensors1 if mask_part == 'tail' else tensors2
                elif name == 'train':
                    tensors = tensors[ : 5000]
                pos_batch = tensors[i * batch : min((i + 1) * batch, tensors.shape[0])]
                dates, dates_id = pos_batch[ : , 3 : 6], pos_batch[ : , 6 : ]
                if eval:
                    facts = pos_batch[ : , : 3].unsqueeze(1)
                else:
                    facts = self.get_pos_neg_batch(pos_batch, mask_part=mask_part, filter=filter)
                
                # print(facts.shape, i, n_batches)
                    
                yield facts, dates, mask_part, dates_id

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
        return triplet[1 : 6] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : 6]

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