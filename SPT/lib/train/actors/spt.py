from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class SPTActor(BaseActor):
    """ Actor for training the multi-modal SPT"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        color_feat_dict_list = []
        depth_feat_dict_list = []
        visiontext_feat_dict_list = []
        feat_dict_list = []
        data['nl_token_ids'] = data['nl_token_ids'].permute(1, 0)
        data['nl_token_masks'] = data['nl_token_masks'].permute(1, 0)
        text_data = NestedTensor(data['nl_token_ids'], data['nl_token_masks'])
        text_dict = self.net(text_data=text_data, mode="language_backbone")
        visiontext_feat_dict_list.append(text_dict)
        # depth_feat_dict_list.append(text_dict)  # depth & language not co

        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_color_i = template_img_i[:,:3,:,:]
            template_depth_i = template_img_i[:,3:,:,:]
            color_template = self.net(img=NestedTensor(template_color_i, template_att_i), mode='backbone_color')
            color_feat_dict_list.append(color_template)
            depth_feat_dict_list.append(self.net(img=NestedTensor(template_depth_i, template_att_i), mode='backbone_depth'))
            visiontext_feat_dict_list.append(color_template)

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        search_color = search_img[:,:3,:,:]
        search_depth = search_img[:,3:,:,:]
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        color_search = self.net(img=NestedTensor(search_color, search_att), mode='backbone_color')
        color_feat_dict_list.append(color_search)
        depth_feat_dict_list.append(self.net(img=NestedTensor(search_depth, search_att), mode='backbone_depth'))
        visiontext_feat_dict_list.append(color_search)

        # run the transformer and compute losses
        seq_dict_color = merge_template_search(color_feat_dict_list)
        seq_dict_depth = merge_template_search(depth_feat_dict_list)
        seq_dict_vl = merge_template_search(visiontext_feat_dict_list)
        out_dict, _, _ = self.net(seq_dict_c=seq_dict_color, seq_dict_d=seq_dict_depth, seq_dict_vl=seq_dict_vl,
                                  mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
