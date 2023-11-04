from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.spt import build_spt
from lib.test.tracker.spt_utils import Preprocessor
from lib.utils.box_ops import clip_box
from pytorch_pretrained_bert import BertTokenizer
from lib.utils.misc import NestedTensor

class SPT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SPT, self).__init__(params)
        network = build_spt(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        print('loading the trained SPT done!'+ self.params.checkpoint)

        vocab_path = self.params.cfg.MODEL.LANGUAGE.VOCAB_PATH

        if vocab_path is not None and os.path.exists(vocab_path):
            self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
            print('loading pretrained Bert done')
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.params.cfg.MODEL.LANGUAGE.TYPE, do_lower_case=True)
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict_color = {}
        self.z_dict_depth = {}

    def initialize(self, image, info: dict):

        # forward the template once
        self.text_input = self._text_input_process(info['init_nlp'], self.params.cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN)
        with torch.no_grad():
            self.text_dict = self.network.forward_text(self.text_input)
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template_color = self.preprocessor.process(z_patch_arr[:,:,:3], z_amask_arr)
        template_depth = self.preprocessor.process(z_patch_arr[:,:,3:], z_amask_arr)

        with torch.no_grad():
            self.z_dict_color = self.network.forward_backbone_color(template_color)
            self.z_dict_depth = self.network.forward_backbone_depth(template_depth)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_color = self.preprocessor.process(x_patch_arr[:,:,:3], x_amask_arr)
        search_depth = self.preprocessor.process(x_patch_arr[:,:,3:], x_amask_arr)
        with torch.no_grad():
            x_dict_color = self.network.forward_backbone_color(search_color)
            x_dict_depth = self.network.forward_backbone_depth(search_depth)
            # merge the template and the search
            feat_dict_list_vl = [self.text_dict, self.z_dict_color, x_dict_color]
            feat_dict_list_color = [self.z_dict_color, x_dict_color]
            feat_dict_list_depth = [self.z_dict_depth, x_dict_depth]
            seq_dict_vl = merge_template_search(feat_dict_list_vl)
            seq_dict_color = merge_template_search(feat_dict_list_color)
            seq_dict_depth = merge_template_search(feat_dict_list_depth)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict_c=seq_dict_color, seq_dict_d=seq_dict_depth,
                                                              seq_dict_vl=seq_dict_vl, run_box_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    def _extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask
    def _text_input_process(self, nlp, seq_length):
        text_ids, text_masks = self._extract_token_from_nlp(nlp, seq_length)
        text_ids = torch.tensor(text_ids).unsqueeze(0).cuda()
        text_masks = torch.tensor(text_masks).unsqueeze(0).cuda()
        return NestedTensor(text_ids, text_masks)

def get_tracker_class():
    return SPT
