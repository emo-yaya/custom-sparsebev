import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox
from .utils import inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA
import torch.distributed as dist

@TRANSFORMER.register_module()
class SparseBEVTransformer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SparseBEVTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.decoder = SparseBEVTransformerDecoder(embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, code_size, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas, bev_queries):
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, mlvl_feats, attn_mask, img_metas, bev_queries)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], init_cfg=None):
        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range

        # params are shared across all decoder layers
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_levels, num_classes, code_size, pc_range=pc_range
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas, bev_queries):
        cls_scores, bbox_preds = [], []

        # calculate time difference according to timestamps
        timestamps = np.array([m['img_timestamp'] for m in img_metas], dtype=np.float64)
        timestamps = np.reshape(timestamps, [query_bbox.shape[0], -1, 6])
        time_diff = timestamps[:, :1, :] - timestamps
        time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
        time_diff = torch.from_numpy(time_diff).to(query_bbox.device)  # [B, F]
        img_metas[0]['time_diff'] = time_diff

        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)  # [B, N, 4, 4]
        img_metas[0]['lidar2img'] = lidar2img

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for lvl, feat in enumerate(bev_queries):
            B, TNGC, H, W, Z = feat.shape  # [B, TN, GC, H, W, Z]
            N, T, G, C = 1, 1, 4, TNGC // 4
            feat = feat.reshape(B, T, N, G, C, H, W, Z)

            feat = feat.reshape(B*T*G, C, N, H, W, Z)
            
            bev_queries[lvl] = feat.contiguous()

        for i in range(self.num_layers):
            DUMP.stage_count = i

            query_feat, cls_score, bbox_pred = self.decoder_layer(
                query_bbox, query_feat, mlvl_feats, attn_mask, img_metas, bev_queries
            )
            query_bbox = bbox_pred.clone().detach()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_levels=4, num_classes=10, code_size=10, num_cls_fcs=2, num_reg_fcs=2, pc_range=[], init_cfg=None):
        super(SparseBEVTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SparseBEVSampling(embed_dims, num_frames=num_frames, num_groups=4, num_points=num_points, num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=4, out_points=128)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)
        
        # self.sampling_bev = CustomBEVSampling(embed_dims, num_frames=1, num_groups=4, num_points=num_points, num_levels=3, pc_range=pc_range)
        self.mixing_bev = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=4, out_points=128)
        self.norm_bev = nn.LayerNorm(embed_dims)

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        # self.sampling_bev.init_weights()
        self.mixing.init_weights()
        self.mixing_bev.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas, bev_feats):
        """
        query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
        """
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))
        sampled_feat, sampled_feat_bev = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas, bev_feats)
        
        query_feat_cam = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat_bev = self.norm_bev(self.mixing_bev(sampled_feat_bev, query_feat))

        fusion_weight = self.fusion_gate(torch.cat([query_feat_cam, query_feat_bev], dim=-1))

        query_feat = fusion_weight * 0 * query_feat_cam + 1 * query_feat_bev

        query_feat = self.norm3(self.ffn(query_feat))

        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)
        
        time_diff = img_metas[0]['time_diff']  # [B, F]
        if time_diff.shape[1] > 1:
            time_diff = time_diff.clone()
            time_diff[time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

        if DUMP.enabled:
            query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            cls_score_sig = torch.sigmoid(cls_score)
            torch.save(query_bbox_dec.cpu(), '{}/query_bbox_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(bbox_pred_dec.cpu(), '{}/bbox_pred_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(cls_score_sig.cpu(), '{}/cls_score_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        return query_feat, cls_score, bbox_pred


class SparseBEVSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


class SparseBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

        self.bev_num_levels = 3
        self.bev_scale_weights = nn.Linear(embed_dims, num_groups * num_points * self.bev_num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_metas, bev_feats):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)
        
        # scale bev weights
        bev_scale_weights = self.bev_scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.bev_num_levels)
        bev_scale_weights = torch.softmax(bev_scale_weights, dim=-1)
        bev_scale_weights = bev_scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.bev_num_levels)

        # sampling
        sampled_feats, sampled_feats_bev = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w,
            bev_feats=bev_feats, scale_weights_bev=bev_scale_weights, pc_range=self.pc_range
        )  # [B, Q, G, FP, C]

        return sampled_feats, sampled_feats_bev

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas, bev_feats):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, bev_feats, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas, bev_feats)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)



class CustomBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""

    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        
        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * 3)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, bev_feats, img_metas):
        '''
        sample bev feat 
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        # _ = bev_feats[0].shape  # feat = feat.reshape(B*T*G, C, N, H, W, Z)

        # 2. 采样点生成
        B, Q = query_bbox.shape[:2]
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]

        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, 1, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)


        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, 3)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, 1, self.num_points, 3)

        sample_points = sampling_points
        B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
        N = 1
        
        sample_points = sample_points.reshape(B, Q, T, G * P, 3)

        # # get the projection matrix
        # lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
        # lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
        # lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

        # expand the points
        ones = torch.ones_like(sample_points[..., :1])
        sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
        sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
        sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
        sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

        # project 3d sampling points to N views
        # sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]
        sample_points = sample_points.squeeze(-1)

        # homo coord -> pixel coord
        # homo = sample_points_cam[..., 2:3]
        # homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + 1e-5)
        # sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
        sampling_points = sample_points[..., 0:3]
        # # normalize
        # sample_points_cam[..., 0] /= image_w
        # sample_points_cam[..., 1] /= image_h
        sampling_points[..., 0] = (sampling_points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        sampling_points[..., 1] = (sampling_points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        sampling_points[..., 2] = (sampling_points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        # check if out of image
        valid_mask = ( 
            (sampling_points[..., 1:2] > 0.0)
            & (sampling_points[..., 1:2] < 1.0)
            & (sampling_points[..., 0:1] > 0.0)
            & (sampling_points[..., 0:1] < 1.0)
            & (sampling_points[..., 2:3] > 0.0)
            & (sampling_points[..., 2:3] < 1.0)
        ).squeeze(-1).float()  # [B, T, N, Q, GP]

        valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
        sampling_points = sampling_points.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

        # prepare batched indexing
        i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
        i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
        i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
        i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
        i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
        i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
        i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
        i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
        
        # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
        i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

        # index the only one sampling point and its valid flag
        sampling_points = sampling_points[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
        valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

        # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
        # sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

        # reorganize the tensor to stack T and G to the batch dim for better parallelism
        sampling_points = sampling_points.reshape(B, T, Q, G, P, 1, 3)
        sampling_points = sampling_points.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
        sampling_points = sampling_points.reshape(B*T*G, Q, P, 3)

        # reorganize the tensor to stack T and G to the batch dim for better parallelism
        scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
        scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
        scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

        sampling_points = sampling_points.contiguous()
        scale_weights = scale_weights.contiguous()

        sampling_locations = sampling_points
        B, C, _, _, _, _ = bev_feats[0].shape
        _, Q, P, _ = sampling_locations.shape

        sampling_locations = sampling_locations * 2 - 1
        sampling_locations = sampling_locations[:, :, :, None, :]  # [B, Q, P, 1, 3]

        final = torch.zeros([B, C, Q, P], device=bev_feats[0].device)

        for lvl, feat in enumerate(bev_feats):
            # feat (B*T*G, C, N, H, W, Z)
            feat = feat.permute(0, 1, 5, 3, 4, 2).squeeze(-1)
            out = F.grid_sample(
                feat, sampling_locations, mode='bilinear',
                padding_mode='zeros', align_corners=True,
            )[..., 0]  # [B, C, Q, P]
            out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
            final += out

        final = final.permute(0, 2, 1, 3)

        # reorganize the sampled features
        BG, Q, C, P = final.shape # [BG, Q, C, P]
        B = BG // G
        final = final.reshape(B, G, Q, C, P)
        final = final.permute(0, 2, 1, 4, 3) # [B, Q, G, P, C]
        
        return final

    def forward(self, query_bbox, query_feat, bev_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, bev_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, bev_feats, img_metas)