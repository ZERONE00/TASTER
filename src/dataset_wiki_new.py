import os
import torch
import math
import numpy as np
from collections import Counter
from datetime import datetime
from datetime import timedelta

class Dataset_Wiki_v2():
    def __init__(self, args):
        self.args = args
        self.name = 'wikidata'
        self.base_dir = os.path.join('../datasets/', self.name)
        self.data = {'train' : None, 'valid' : None, 'test' : None}
        self.fact_count = args.fact_count
        self.ent2id, self.rel2id, self.date2id = {}, {}, {}
        self.process_raw_data()
        self.build_dicts()
        self.build_train_data()
        self.build_eval_data()
        self.get_eval_mask_dict()
        self.sample_size = args.sample_size - 1
        self.nentity, self.nrelation = len(self.ent2id) + 1, len(self.rel2id)
        print('nentity : {}, ntimes : {}'.format(self.nentity, len(self.date_class)))
        self.get_mask_dict()
        self.min_year = 0
        

    def process_raw_data(self):
        # each item in self.raw_data[key] has the form of (head, rel, tail, begin, end)
        def _process_date(date):
            try:
                return int(date[:4])
            except Exception as e:
                # print('waring: An exception was thrown while processing date {}'.format(e))
                return None

        self.raw_data = {'train' : None, 'valid' : None, 'test' : None}
        for key in ["train", "valid", "test"]:
            path = os.path.join(self.base_dir, '{}.txt'.format(key))
            triplets = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    h, r, t, d1, d2 = line.split('\t')
                    h, t, r = h.strip(), t.strip(), r.strip()
                    d1 = _process_date(d1)
                    d2 = _process_date(d2)
                    if d1 == None and d2 == None:
                        d1, d2 = 0, 0
                    elif d1 == None:
                        d1 = d2
                    elif d2 == None:
                        d2 = d1
                    triplets.append((h, r, t, d1, d2))
            self.raw_data[key] = triplets

    def get_split_times(self, d1, d2, step=1):
        # split [d1, d2] to [d1, d1+step], [d1+step, d1+step*2],..., [d2-step, d2], add d1, d1+step,..,d2 to split_times
        split_times = [d1, d2]
        start = d1 + step
        while start < d2:
            split_times.append(start)
            start = start + step
        split_times = list(set(split_times))
        return split_times

    def todateid(self, date):
        left, right = 0, len(self.date_class) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.date_class[mid] < date:
                left = mid + 1
            else:
                right = mid - 1
        
        return left

    def process_time(self):
        dates, self.date_class = [], []
        for key in ["train", "valid", "test"]:
            for triple in self.raw_data[key]:
                _, _, _, d1, d2 = triple
                dates = dates + self.get_split_times(d1, d2)
            
        dates.sort()
        self.freq, count = Counter(dates), 0
        self.inverse = int(100 / self.args.step)

        for key in sorted(self.freq.keys()):
            count += self.freq[key]
            if count >= self.fact_count:
                self.date_class.append(key)
                count=0

        if key != self.date_class[-1]:
            self.date_class.append(key)

        # for i, yc in enumerate(self.date_class): 
        #     self.date2id[yc] = i 

    def get_id(self, key, idmap):
        if key in idmap:
            return idmap[key]
        idmap[key] = len(idmap)
        return idmap[key]

    def build_dicts(self):
        # build date2id, ent2id, rel2id
        self.process_time()
        for key in ["train", "valid", "test"]:
            for triple in self.raw_data[key]:
                h, r, t, d1, d2 = triple
                split_times = self.get_split_times(d1, d2)
                for date in split_times:
                    self.date2id[date] = self.todateid(date)

                self.get_id(h, self.ent2id)
                self.get_id(t, self.ent2id)
                self.get_id(r, self.rel2id)

    def build_split_data(self, key='train', step=1):
        split_data = []
        if key not in self.raw_data:
            print('err: there exist no key named {} in our data'.format(key))
            exit()
        for triple in self.raw_data[key]:
            h, r, t, d1, d2 = triple
            h = self.get_id(h, self.ent2id)
            t = self.get_id(t, self.ent2id)
            r = self.get_id(r, self.rel2id)
            split_times = self.get_split_times(d1, d2, step)
            for date in split_times:
                d_id = self.date2id[date]
                split_data.append((h, r, t, date, d_id))
        return split_data

    def build_interval_data(self, key='test'):
        if key not in self.raw_data:
            print('err: there exist no key named {} in our data'.format(key))
            exit()
        interval_data = []
        for triple in self.raw_data[key]:
            h, r, t, d1, d2 = triple
            h = self.get_id(h, self.ent2id)
            t = self.get_id(t, self.ent2id)
            r = self.get_id(r, self.rel2id)
            # d1 = self.date2id[d1]
            # d2 = self.date2id[d2]
            interval_data.append((h, r, t, d1, d2))
        return interval_data
            
    def build_train_data(self):
        self.train_data = self.build_split_data('train', step=self.args.step)

    def build_eval_data(self):
        self.eval_data = {}
        for key in self.data:
            self.eval_data[key] = self.build_interval_data(key)

    def iterate_eval(self, key='train'):
        # batch = 1
        triples = self.eval_data[key]
        if key == 'train':
            triples = triples[:1000]
        # n_batches = math.ceil(tensors.shape[0] / batch)
        fs = ['head', 'tail'] 
        for triple in triples:
            for mask_part in fs:
                h, r, t, d1, d2 = triple
                split_times = self.get_split_times(d1, d2, self.inverse)
                # split_tensor = torch.tnesor[len(split_times), 5]
                split_tensor = []
                for date in split_times:
                    d_id = self.date2id[date]
                    split_tensor.append((h, r, t, date, d_id))
                split_tensor = torch.tensor(split_tensor)

                dates, dates_id = split_tensor[ : , 3 : 4], split_tensor[ : , 4 : ]
                facts = split_tensor[ : , : 3].unsqueeze(1)
                yield facts, dates, mask_part, dates_id, triple

    def iterate_train(self, filter=True):
        batch = self.args.bsize
        tensors = torch.tensor(self.train_data)
        tensors1 = tensors[torch.randperm(tensors.shape[0])] # shuffle
        tensors2 = tensors[torch.randperm(tensors.shape[0])] # shuffle
        n_batches = math.ceil(tensors.shape[0] / batch)
        fs = ['head', 'tail']
        for i in range(n_batches):
            for mask_part in fs:
                tensors = tensors1 if mask_part == 'tail' else tensors2
                pos_batch = tensors[i * batch : min((i + 1) * batch, tensors.shape[0])]
                dates, dates_id = pos_batch[ : , 3 : 4], pos_batch[ : , 4 : ]
                facts = self.get_pos_neg_batch(pos_batch, mask_part=mask_part, filter=filter)
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
    

    def get_mask_key(self, triplet, mask_part='head'):
        return triplet[1 : 4] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : 4]

    def get_mask_dict(self):
        self.mask_dict = {'head' : {}, 'tail' : {}}
        for key in self.data:
        # for key in ['train']:
            split_data = self.build_split_data(key)
            for triplet in split_data:
                head_mask, tail_mask = self.get_mask_key(triplet), self.get_mask_key(triplet, mask_part='tail')
                if head_mask not in self.mask_dict['head']:
                    self.mask_dict['head'][head_mask] = np.array([]).astype(int)
                tmp = self.mask_dict['head'][head_mask]
                self.mask_dict['head'][head_mask] = np.append(tmp, triplet[0])

                if tail_mask not in self.mask_dict['tail']:
                    self.mask_dict['tail'][tail_mask] = np.array([]).astype(int)
                tmp = self.mask_dict['tail'][tail_mask]
                self.mask_dict['tail'][tail_mask] = np.append(tmp, triplet[2])

    def get_eval_mask_key(self, triplet, mask_part='head'):
        return triplet[1 : 5] if mask_part == 'head' else triplet[0 : 2] + triplet[3 : 5]

    def get_eval_mask_dict(self):
        self.eavl_mask_dict = {'head' : {}, 'tail' : {}}
        for key in self.data:
            interval_data = self.build_interval_data(key)
            for triplet in interval_data:
                head_mask, tail_mask = self.get_eval_mask_key(triplet), self.get_eval_mask_key(triplet, mask_part='tail')
                if head_mask not in self.eavl_mask_dict['head']:
                    self.eavl_mask_dict['head'][head_mask] = np.array([]).astype(int)
                tmp = self.eavl_mask_dict['head'][head_mask]
                self.eavl_mask_dict['head'][head_mask] = np.append(tmp, triplet[0])

                if tail_mask not in self.eavl_mask_dict['tail']:
                    self.eavl_mask_dict['tail'][tail_mask] = np.array([]).astype(int)
                tmp = self.eavl_mask_dict['tail'][tail_mask]
                self.eavl_mask_dict['tail'][tail_mask] = np.append(tmp, triplet[2])
