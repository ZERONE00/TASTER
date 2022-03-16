import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TASTER(nn.Module):
    def __init__(self, args):
        super(TASTER, self).__init__()
        self.args = args
        self.dim = self.args.dim * 2
        self.rel_embeds_g = nn.Embedding(args.nrelation, self.dim)
        self.ent_embeds = nn.Embedding(args.nentity, self.dim)
        self.rel_embeds = nn.Embedding(args.nrelation, self.dim // 2)
        self.radio = torch.rand(args.nrelation)
        self.radio = torch.nn.Parameter(self.radio, requires_grad=True)

        nn.init.uniform_(self.ent_embeds.weight, -self.args.init1, self.args.init1)
        nn.init.uniform_(self.rel_embeds.weight, -self.args.init2, self.args.init2)

        if self.args.shear:
            self.shear_embeds = nn.Embedding(args.n_day, self.args.dim * 2).cuda()
            nn.init.uniform_(self.shear_embeds.weight, -self.args.init3, self.args.init3)

        elif self.args.block:
            # use block matrix
            self.block_size = args.block_size * 2
            if self.dim % self.block_size != 0:
                raise ValueError('dim have to divide block_num')
            self.block_num = self.dim // self.block_size
            self.trans_matrix = nn.Parameter(torch.Tensor(args.n_day, self.block_num * self.block_size * self.block_size))
            self.eye_block = torch.eye(self.block_size)
            self.mask_block = (self.eye_block < 0.5)
            if True:
                self.eye_block = self.eye_block.cuda()
                self.mask_block = self.mask_block.cuda()
            nn.init.uniform_(self.trans_matrix, -self.args.init3, self.args.init3)

        elif self.args.fix_pattern:
            # use fixed unstructured sparse matrix
            self.block_size = args.block_size * 2 - 1 if self.args.diagonal else args.block_size * 2
            ids = self.get_sparse_ids()
            self.sparse_matrices = []
            for i in range(args.n_day):
                value = (torch.rand(self.dim * self.block_size) - 0.5) * 2 * args.init3
                matrix = torch.sparse.FloatTensor(ids, value, torch.Size([self.dim, self.dim])).to_dense()
                self.sparse_matrices.append(matrix)
        else:
            # use unfixed unstructured sparse matrix
            # self.args.sparse_rate = args.sparse_rate - 1 if self.diagonal else args.sparse_rate
            self.args.sparse_rate = args.sparse_rate
            for i in range(args.n_day):
                ids = self.get_sparse_ids()
                value = (torch.rand(self.dim * self.args.sparse_rate) - 0.5) * 2 * args.init3
                matrix = torch.sparse.FloatTensor(ids, value, torch.Size([self.dim, self.dim])).to_dense()
                self.sparse_matrices.append(matrix)

        if not self.args.block and not self.args.shear:
            self.sparse_matrices = torch.stack(self.sparse_matrices, dim=0)
            self.masks = (self.sparse_matrices != 0).cuda()
            self.sparse_matrices = torch.nn.Parameter(self.sparse_matrices, requires_grad=True)

    def get_s_score(self, heads, rels, tails):
        ori = self.ent_embeds(heads).view(-1, heads.shape[1], self.dim // 2, 2)
        tar = self.ent_embeds(tails).view(-1, tails.shape[1], self.dim // 2, 2)
        h1, h2 = ori[..., 0], ori[..., 1]
        t1, t2 = tar[..., 0], tar[..., 1]
        
        s_scores = torch.cat((h2, h1), -1) + self.rel_embeds_g(rels) - torch.cat((t1, t2), -1)
        s_scores = F.dropout(s_scores, p=self.args.dropout, training=self.training)
        s_scores = -torch.norm(s_scores, dim=-1, p=1)
        return s_scores
    
    def get_t_score(self, ori, tar, rels, mask_part='tail'):
        ori = ori.view(-1, ori.shape[1], self.dim // 2, 2)
        tar = tar.view(-1, tar.shape[1], self.dim // 2, 2)
        ori_rel, ori_img = ori[..., 0], ori[..., 1]
        tar_rel, tar_img = tar[..., 0], tar[..., 1]
        # rels = rels[:, 0]
        # rels = rels.unsqueeze(1)
        relations = self.rel_embeds(rels)
        relations = relations / (self.args.init2 / np.pi)
        cos_rotation, sin_rotation = torch.cos(relations), torch.sin(relations) # cos(\theta), sin(\theta)
        
        if mask_part == 'tail':
            rot_rel = ori_rel * cos_rotation + ori_img * sin_rotation # x * cos(\theta) + y * sin(\theta)
            rot_img = ori_img * cos_rotation - ori_rel * sin_rotation # -x * sin(\theta) + y * cos(\theta)
        else:
            rot_rel = ori_rel * cos_rotation - ori_img * sin_rotation # x * cos(-\theta) + y * sin(-\theta)
            rot_img = ori_img * cos_rotation + ori_rel * sin_rotation # -x * sin(-\theta) + y * cos(-\theta)
        
        re_score = rot_rel - tar_rel
        im_score = rot_img - tar_img
        scores = torch.stack([re_score, im_score], dim = 0)
        scores = torch.norm(scores, dim=0, p=2)
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = -torch.sum(scores, dim = -1)

        return scores

    def get_rotation_matrix(self, rels, mask_part='tail'):
        relations = self.rel_embeds(rels) / (self.args.init2 / np.pi)
        if mask_part == 'head':
            relations = -relations
        cos_rotation, sin_rotation = torch.cos(relations), torch.sin(relations)
        rotation_matrix = torch.stack((cos_rotation, -sin_rotation, sin_rotation, cos_rotation), dim=-1)
        return rotation_matrix.view(-1, self.dim // 2, 2, 2)
    
    def get_t_score_2(self, ori, tar, rels, mask_part='tail'):
        rels = rels[:, 0]
        rotation_matrix = self.get_rotation_matrix(rels, mask_part)
        sample = ori.shape[1]
        ori = ori.view(-1, sample, self.dim // 2, 2)
        # print(rotation_matrix.shape, ori.shape)
        rot = torch.einsum('ijkm, iljm -> iljk', rotation_matrix, ori)

        scores = rot - tar.view(-1, sample, self.dim // 2, 2)
        scores = torch.norm(scores, dim=-1, p=2)
        
        scores = F.dropout(scores, p=self.args.dropout, training=self.training)
        scores = -torch.sum(scores, dim = -1)
        # print(scores.shape)
        return scores

    def get_sparse_ids(self):
        # sparse_rate = self.args.sparse_rate - 1 if self.args.diagonal else self.args.sparse_rate
        a, b = [], []
        self.args.sparse_rate = self.args.block_size * 2 - 1 if self.args.diagonal else self.args.block_size * 2
        for i in range(self.args.sparse_rate):
            a.append(torch.randperm(self.args.dim * 2))
            b.append(torch.randperm(self.args.dim * 2))
        a, b = torch.cat(a), torch.cat(b)
        ids = torch.stack((a,b) ,dim = 0)
        return ids

    def transform_by_sparse_matrix(self, date_ids, embeds):
        trans_matrix = self.sparse_matrices[date_ids].squeeze(1) * self.masks[date_ids].squeeze(1)
        # print(trans_matrix.shape, embeds.shape)
        # embeds_trans = torch.matmul(trans_matrix, embeds)
        embeds_trans = torch.einsum('ijk, ilk -> ilj', trans_matrix, embeds)
        if self.args.diagonal:
            embeds_trans += embeds
        return embeds_trans
    
    def transform_by_block_matrix(self, date_ids, embeds):
        # print(self.trans_matrix[date_ids].shape, embeds.shape)
        trans_matrix = self.trans_matrix[date_ids].view(-1, self.block_num, self.block_size, self.block_size)
        if self.args.diagonal:
            # set diagonal one
            trans_matrix = trans_matrix * self.mask_block + self.eye_block

        embeds = embeds.view(-1, embeds.shape[1], self.block_num, self.block_size)
        
        embeds = torch.einsum('ijkm, iljm -> iljk', trans_matrix, embeds)

        return embeds

    def transform_by_shearing(self, date_ids, embeds):
        embeds = embeds.view(-1, embeds.shape[1], self.dim // 2, 2)
        embeds_rel, embeds_img = embeds[..., 0], embeds[..., 1]
        a, b = torch.chunk(self.shear_embeds(date_ids), 2, dim=2)
        embeds_rel, embeds_img = embeds_rel + a * embeds_img, embeds_img + b * embeds_rel
        return torch.cat((embeds_rel, embeds_img), dim=-1)
    
    def forward(self, heads, rels, tails, date_ids, mask_part='tail'):
        if not self.args.local:
            s_scores = self.get_s_score(heads, rels, tails)
        else:
            s_scores = 0
        
        emb_head = self.ent_embeds(heads)
        emb_tail = self.ent_embeds(tails)

        if self.args.shear:
            emb_head = self.transform_by_shearing(date_ids, emb_head)
            emb_tail = self.transform_by_shearing(date_ids, emb_tail)
        elif self.args.block:
            emb_head = self.transform_by_block_matrix(date_ids, emb_head)
            emb_tail = self.transform_by_block_matrix(date_ids, emb_tail)
        else:
            emb_head = self.transform_by_sparse_matrix(date_ids, emb_head)
            emb_tail = self.transform_by_sparse_matrix(date_ids, emb_tail)
        
        if mask_part == 'tail':
            # rotate from head to tail
            ori = emb_head
            tar = emb_tail
        else:
            ori = emb_tail
            tar = emb_head

        t_scores = self.get_t_score_2(ori, tar, rels, mask_part=mask_part)

        if self.args.local:
            return t_scores
        
        radioes = torch.sigmoid(self.radio[rels[:, 0]]).unsqueeze(1)
        
        scores = s_scores * radioes + t_scores * (1 - radioes)
        # print(scores.shape)
        return scores

    def reg_loss(self):
        if self.args.shear:
            return self.args.reg1 * torch.norm(self.shear_embeds.weight, p=1)
        elif self.args.block:
            if self.args.diagonal:
                trans_matrix = self.trans_matrix.view(-1, self.block_num, self.block_size, self.block_size)
                trans_matrix = trans_matrix * self.mask_block
            else:
                trans_matrix = self.trans_matrix

            return self.args.reg1 * (
                                # torch.norm(self.ent_embeds.weight)
                                # + torch.norm(self.rel_embeds_g.weight) +
                                 torch.norm(trans_matrix, p=1)
                                )
        else:
            return self.args.reg1 * (
                                # torch.norm(self.ent_embeds.weight)
                                # + torch.norm(self.rel_embeds_g.weight)
                                torch.norm(self.sparse_matrices * self.masks, p=1)
                                )

    def weight_loss(self):
        return self.args.reg2 * (
                                torch.norm(self.ent_embeds.weight) ** 2
                                # + torch.norm(self.rel_embeds_g.weight) ** 2
                                #  torch.norm(trans_matrix, p=1)
                                )
    
    def show_trans(self):
        if self.args.diagonal:
            trans_matrix = self.trans_matrix.view(-1, self.block_num, self.block_size, self.block_size)
            trans_matrix = trans_matrix * self.mask_block
            print("mean trans: {:.4f}".
                    format(torch.mean(trans_matrix).item()))
        else:
            trans_matrix = self.trans_matrix
            print("mean trans: {:.4f}".
                    format(torch.mean(trans_matrix).item()))
        

    def get_embed_at_t(self, ent_id, time_id=None):
        with torch.no_grad():
            ent_id = torch.LongTensor([ent_id]).cuda().unsqueeze(0)
            global_embed = self.ent_embeds(ent_id)

            if not time_id:
                return global_embed

            time_id = torch.LongTensor([time_id]).cuda()
            
            # print(global_embed.shape)
            if self.args.shear:
                embed_at_t = self.transform_by_shearing(time_id, global_embed)
            elif self.args.block:
                embed_at_t = self.transform_by_block_matrix(time_id, global_embed)
            else:
                embed_at_t = self.transform_by_sparse_matrix(time_id, global_embed)
            
        return embed_at_t
