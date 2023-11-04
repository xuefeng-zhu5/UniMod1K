import os.path

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class UniMod1KDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.unimod1k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        nlp_path = '{}/{}/nlp.txt'.format(self.base_path, sequence_name)
        nlp_label = load_text(str(nlp_path), delimiter=',', dtype=str)
        nlp_label = str(nlp_label)

        end_frame = ground_truth_rect.shape[0]

        depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                        sequence_path=sequence_path, frame=frame_num, nz=nz)
                        for frame_num in range(start_frame, end_frame+1)]
        color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                        sequence_path=sequence_path, frame=frame_num, nz=nz)
                        for frame_num in range(start_frame, end_frame+1)]

        frames = []
        for c_path, d_path in zip(color_frames, depth_frames):
            frames.append({'color': c_path, 'depth': d_path})

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'unimod1k', ground_truth_rect, language_query=nlp_label)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_file = os.path.join(self.base_path, 'list.txt')
        with open(list_file, 'r') as f:
            sequence_list = f.read().splitlines()
        return sequence_list