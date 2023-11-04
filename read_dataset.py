import os
import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from SPT.lib.train.dataset.base_video_dataset import BaseVideoDataset
from SPT.lib.train.data import jpeg4py_loader
from SPT.lib.train.admin import env_settings
import cv2
import argparse
from SPT.lib.train.dataset.depth_utils import get_rgbd_frame
import matplotlib

# UniMod1K Class
class UniMod1K(BaseVideoDataset):
    def __init__(self, root=None, nlp_root=None,  dtype='rgbcolormap', image_loader=jpeg4py_loader):
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        super().__init__('UniMod1K', root, image_loader)
        self.root_nlp = nlp_root
        self.root = root
        self.dtype = dtype
        self.sequence_list = self._build_sequence_list()

        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

    def _build_sequence_list(self):

        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SPT/lib/train')
        file_path = os.path.join(ltr_path, 'data_specs', 'unimod1k_train_split.txt')
        sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('/')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return 'unimod1k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)


    def _get_sequence_path(self, seq_id):
        '''
        Return :
                - seq path
        '''
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def _get_nlp_path(self, seq_id):
        '''
        Return :
                - nlp path
        '''
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root_nlp, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        nlp_path = self._get_nlp_path(seq_id)
        '''
        if the box is too small, it will be ignored
        '''
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid
        nlp = self._read_nlp(nlp_path)

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'nlp': nlp}

    def _read_nlp(self, nlp_path):
        nlp_file = os.path.join(nlp_path, "nlp.txt")
        nlp = ""
        try:
            nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
        except Exception as e:
            print(e)
            print(f'nlp_file:{nlp_file}')
        return nlp[0][0]

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp = self._read_nlp(seq_path)
        return nlp

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return rgb depth image path
        '''
        return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id + 1)), os.path.join(seq_path, 'depth',
                                                                                               '{:08}.png'.format(
                                                                                                   frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id, bbox=None):
        '''
        Return :
            - rgb
            - colormap from depth image

        '''
        color_path, depth_path = self._get_frame_path(seq_path, frame_id)

        img = get_rgbd_frame(color_path, depth_path, dtype=self.dtype, depth_clip=False)

        return img

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        depth_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(depth_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        depth_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(depth_path)
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        frame_list = [self._get_frame(depth_path, f_id, bbox=anno_frames['bbox'][ii]) for ii, f_id in
                      enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

def main():
    parser = argparse.ArgumentParser(description='Read images and labels.')
    parser.add_argument('--data_dir', type=str, help='Name of tracking method.')
    parser.add_argument('--nlp_dir', type=str, help='Name of config file.')
    parser.add_argument('--seq_id', default=0, type=int, help='Name of config file.')
    args = parser.parse_args()

    unimod1k = UniMod1K(args.data_dir, args.nlp_dir)
    frame_id = [0]

    rgbd_img, annos, _ = unimod1k.get_frames(args.seq_id, frame_id)
    rgb = rgbd_img[0][:, :, :3]
    depth_color = cv2.cvtColor(rgbd_img[0][:, :, 3:], cv2.COLOR_BGR2RGB)  # converted to colormap for visualisation
    gt_bbox = list(map(int, np.array(annos['bbox'][0]).tolist()))
    nlp = annos['nlp'][0]
    im_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.rectangle(im_vis, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2],gt_bbox[1] + gt_bbox[3]), (0, 0, 255), 3)
    cv2.putText(im_vis, nlp, (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.rectangle(depth_color, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2],gt_bbox[1] + gt_bbox[3]), (0, 255, 255), 3)
    plt.subplot(211)
    plt.imshow(im_vis[:,:, ::-1])
    plt.subplot(212)
    plt.imshow(depth_color)
    plt.show()

if __name__ == '__main__':
    main()