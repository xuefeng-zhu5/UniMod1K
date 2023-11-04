"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import os
from ..language_model import build_bert
from .utils import _init_weights_xavier, _init_weights_trunc_normal
import collections

class SPT(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, cfg, backbone_color, backbone_depth, language_backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone_color = backbone_color
        self.backbone_depth = backbone_depth
        self.language_backbone = language_backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck_color = nn.Conv2d(backbone_color.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.bottleneck_depth = nn.Conv2d(backbone_depth.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.text_proj = nn.Linear(self.language_backbone.num_channels, hidden_dim)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.divisor = 16
        self.num_visu_template_token = int((cfg.DATA.TEMPLATE.SIZE // self.divisor) ** 2)
        self.num_visu_search_token = int((cfg.DATA.SEARCH.SIZE // self.divisor) ** 2)
        self.num_text_token = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN
        self.nl_pos_embed = nn.Embedding(self.num_text_token, hidden_dim)
        self.num_total = self.num_visu_template_token + self.num_visu_search_token # + self.num_text_token
        self.init = cfg.MODEL.INIT
        self._reset_parameters(self.init)

    def _reset_parameters(self, init):
        # parameters init
        if init == 'xavier':
            self._init_weights_func = _init_weights_xavier
        elif init == 'trunc_norm':
            self._init_weights_func = _init_weights_trunc_normal
        else:
            raise RuntimeError(F"init method should be xavier/trunc_norm, not {init}.")
        self.nl_pos_embed.apply(self._init_weights_func)
        self.text_proj.apply(self._init_weights_func)


    def forward(self, img=None, seq_dict_c=None, seq_dict_d=None, seq_dict_vl=None, text_data=None, mode="backbone",
                run_box_head=True, run_cls_head=False):
        if mode == "backbone_color":
            return self.forward_backbone_color(img)
        if mode == "backbone_depth":
            return self.forward_backbone_depth(img)
        elif mode == "language_backbone":
            return self.forward_text(text_data)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict_c, seq_dict_d, seq_dict_vl, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone_color(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched color images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone_color(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust_color(output_back, pos)

    def forward_backbone_depth(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched depth images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone_depth(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust_depth(output_back, pos)

    def forward_text(self, text_data: NestedTensor):
        # language bert
        text_fea = self.language_backbone(text_data)
        text_src, text_mask = text_fea.decompose() # seq_len * b * HIDDEN_DIM , seq_len * b
        bs = text_src.shape[0]
        text_src = self.text_proj(text_src).permute(1, 0, 2)
        pos = self.nl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        return {"feat": text_src, "mask": text_mask, "pos": pos}

    def forward_transformer(self, seq_dict_c, seq_dict_d, seq_dict_vl, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        output_embed, enc_mem = self.transformer(seq_dict_c["feat"], seq_dict_c["mask"], seq_dict_c["pos"],
                                                 seq_dict_d["feat"], seq_dict_d["mask"], seq_dict_d["pos"],
                                                 seq_dict_vl, self.query_embed.weight, return_encoder_output=True)
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, enc_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord

    def adjust_color(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck_color(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    def adjust_depth(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck_depth(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_spt(cfg):
    backbone_color = build_backbone(cfg)  # backbone and positional encoding are built together
    backbone_depth = build_backbone(cfg)
    language_backbone = build_bert(cfg)
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    model = SPT(
        cfg,
        backbone_color,
        backbone_depth,
        language_backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    # load encoder and decoder parameters for stark to color, depth, vl encoder
    original_stark_dic = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
    stark_backbone = collections.OrderedDict()
    stark_encoder = collections.OrderedDict()
    stark_decoder = collections.OrderedDict()
    stark_head = collections.OrderedDict()
    stark_neck = collections.OrderedDict()
    stark_query = collections.OrderedDict()

    for key, val in original_stark_dic['net'].items():
        if 'backbone' in key:
            backbone_key = key[9:]
            stark_backbone[backbone_key] = val

        if 'transformer.encoder' in key:
            encoder_key = key[20:]
            stark_encoder[encoder_key] = val

        if 'transformer.decoder' in key:
            decoder_key = key[20:]
            stark_decoder[decoder_key] = val

        if 'box_head' in key:
            head_key = key[9:]
            stark_head[head_key] = val

        if 'bottleneck' in key:
            head_key = key[11:]
            stark_neck[head_key] = val

        if 'query_embed' in key:
            query_key = key[12:]
            stark_query[query_key] = val

    model.backbone_color.load_state_dict(stark_backbone)
    model.backbone_depth.load_state_dict(stark_backbone)
    model.transformer.encoder_color.load_state_dict(stark_encoder)
    model.transformer.encoder_depth.load_state_dict(stark_encoder) #
    model.transformer.encoder_vl.load_state_dict(stark_encoder)
    model.transformer.decoder.load_state_dict(stark_decoder)
    model.box_head.load_state_dict(stark_head)
    model.bottleneck_color.load_state_dict(stark_neck)
    model.bottleneck_depth.load_state_dict(stark_neck)
    model.query_embed.load_state_dict(stark_query)

    return model
