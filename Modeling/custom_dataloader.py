import os
import json
import numpy as np
from torch.utils.data import Dataset
from dataset import keypoints17_to_coco18
from utils.data_utils import trans_list
from dataset import PoseSegDataset
import torch
import math
from utils.data_utils import normalize_pose
from torch.utils.data import DataLoader


class PoseDatasetWithAugmentation(Dataset):
    def __init__(self, path_to_json_dir, path_to_annotation_file, trans_list, evaluate=False, normalize_pose=True, **datast_args):
        self.json_dir = path_to_json_dir
        self.trans_list = trans_list
        self.evaluate = evaluate
        self.normalize_pose = normalize_pose
        self.annotations = path_to_annotation_file
        self.args = datast_args
        
        self.num_transform = (len(trans_list) if not evaluate else 1)
        self.apply_transforms = not evaluate

        self.pose_data = []   
        self.labels = []      
        self.filenames = []   # store full filenames
        
        with open(self.annotations, 'r') as an:
            annotations = json.load(an)

        # --- Scan folder structure and load pose data ---
        for fname in sorted(os.listdir(path_to_json_dir)):
            if not fname.endswith(".json"):
                continue

            # Extract clip_id and video_id from filename
            filename_wo_ext = os.path.splitext(fname)[0]
            video_id, clip_id = filename_wo_ext.rsplit('_', 1)
            json_path = os.path.join(path_to_json_dir, fname)

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Load pose sequences directly (no metadata, no person_sequences)
            for person_id in data.keys():
                pose_seq, _ = self._load_person_sequence(json_path, person_id)

                # If input is [3, T, 17], convert to COCO18 format
                if pose_seq.shape[0] == 3 and pose_seq.shape[2] == 17:
                    pose_seq = keypoints17_to_coco18(
                        pose_seq.transpose(1, 2, 0)
                    ).transpose(2, 0, 1)

                self.pose_data.append(pose_seq)

                s = video_id.replace("padded_", "")

                # Mapping the labels
                if annotations[f"{s}/{clip_id}"][person_id]["class"] in [1, 2]:
                    self.labels.append(0)  # normal
                else:
                    self.labels.append(1)  # abnormal

                # Save filename as: {clip_id}_{video_id}_p{person_id}
                self.filenames.append(f"{clip_id}_{video_id}_p{person_id}")

        self.num_samples = len(self.pose_data)

    def _load_person_sequence(self, json_path, person_id):
        with open(json_path, 'r') as f:
            data = json.load(f)
        person_data = data.get(person_id, {})
        frame_ids = sorted(person_data.keys(), key=lambda x: int(x))
        pose_seq = []
        for frame_id in frame_ids:
            kp = person_data[frame_id].get('keypoints', [])
            assert kp != [], "kp is an empty list! check _load_person_sequence"
            assert len(kp) % 3 == 0, "keypoints are not correct! check _load_person_sequence"
            pose_seq.append(np.array(kp).reshape(-1, 3))
        if not pose_seq:
            raise ValueError(f"No valid frames for person {person_id} in {json_path}")
        pose_seq = np.stack(pose_seq, axis=0)  # [T, V, 3]
        pose_seq = pose_seq.transpose(2, 0, 1)  # [C=3, T, V]
        assert pose_seq.shape == (3, 30, 17), "not valid keypoints! check _load_person_json"
        return pose_seq, frame_ids

    def __len__(self):
        return self.num_samples * self.num_transform

    def __getitem__(self, idx):
        if self.apply_transforms:
            sample_index = idx % self.num_samples
            trans_index = math.floor(idx / self.num_samples)
            data_numpy = np.array(self.pose_data[sample_index])
            data_transformed = self.trans_list[trans_index](data_numpy)
        else:
            sample_index = idx
            data_transformed = np.array(self.pose_data[idx])
            trans_index = 0  # No transformations   

        if self.normalize_pose:
            data_transformed = normalize_pose(
                data_transformed.transpose((1, 2, 0))[None, ...],
                **self.args
            ).squeeze(axis=0).transpose(2, 0, 1)

        T = data_transformed.shape[1]
        assert T == 30, f"the shape of pose is not 30!!! it is {T}"

        label = self.labels[sample_index]
        filename = self.filenames[sample_index]

        pose = torch.from_numpy(data_transformed).float()  
        label = torch.tensor(label, dtype=torch.long)
        scores = torch.zeros(30, dtype=torch.float64)

        return pose, scores, label, filename




def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'headless': args.headless, 'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, "dataset": args.dataset,
                    'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip}
                    
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    for split in splits:
        evaluate = split == 'test'
        abnormal_train_path = args.pose_path_train_abnormal if split == 'train' else None
        normalize_pose_segs = args.global_pose_segs
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset_args['vid_path'] = args.vid_path[split]
        if not evaluate :
            annotations_path="/home/keypoints_annotation/Train_data_results/RGB_annotation.json"
            dataset[split] = PoseDatasetWithAugmentation(args.pose_path[split], 
                                evaluate=evaluate,
                                path_to_annotation_file=annotations_path,
                                 trans_list= dataset_args['trans_list'])

        else :
            annotations_path="/home/keypoints_annotation/Test_data_results/RGB_annotation.json"
            dataset[split] = PoseDatasetWithAugmentation(args.pose_path[split], 
                                evaluate=evaluate,
                                path_to_annotation_file=annotations_path,
                                 trans_list= dataset_args['trans_list'])
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader

    