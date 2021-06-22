import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import pandas as pd

root = os.path.join('D:', 'Countix')


class Countix(Dataset):

    def __init__(self, root, split, clip_len=16, resize_height = 224, resize_width = 224):
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.annotations = self.read_split()
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.debug = False

        if self.split is not 'test':
            self.class_dict = self.generate_class_ids()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        frames = self.get_relevant_clip(index)
        # TODO Spatial crop
        # TODO Normalization
        frames = self.to_tensor(frames)
        return frames, self.annotations['count'][index]
        # Use the return statement below to return video classification instead of frame count
        # return frames, self.class_dict[self.annotations['class'][index]]

    def generate_class_ids(self):
        class_dict = {}
        id = 0
        classes = pd.read_csv(os.path.join(self.root, "CSVs", "countix_class_ids.csv"), sep=",")
        for name in classes['class']:
            class_dict.update({name:id})
            id += 1
        return class_dict

    @staticmethod
    def to_tensor(frames):
        # (clip_len, height, width, channels)
        frames = frames.transpose((3, 0, 1, 2))
        # (channels, clip_len, height, width)
        return torch.from_numpy(frames)

    # Trims clip to start and end points specified in dataset
    def get_relevant_clip(self, index):
        frames, fps = self.read_video(index)

        if self.debug:
            print('FPS = ' + str(fps))
            print(frames.shape)

        start_index = int(fps * self.annotations['repetition_start'][index])
        # TODO Implement variable clip length required for repetition counting
        # Static lengths are fine for classification
        end_index = int(fps * self.annotations['repetition_end'][index])

        if self.debug:
            print(start_index)
            print(end_index)
        return frames[start_index: end_index]

    def read_split(self):
        # Remove '_no_missing_ids' if applicable
        return pd.read_csv(os.path.join(self.root, "CSVs", "Countix_" + self.split + "_no_missing_ids.csv"), sep=",")

    def read_video(self, index):
        """Read video from file."""
        if self.debug:
            print('Video ID = ' + self.annotations['video_id'][index])

        cap = cv2.VideoCapture(os.path.join(self.root, self.split, self.annotations['video_id'][index] + '.mp4'))
        if not cap.isOpened():
            cap = cv2.VideoCapture(os.path.join(self.root, self.split, self.annotations['video_id'][index] + '.webm'))
        assert cap.isOpened(), 'Unable to open video in path ' + os.path.join(self.root, self.split,
                                                                              self.annotations['video_id'][index])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        if cap.isOpened():
            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.resize_width, self.resize_height))
                frames.append(frame_rgb)
        frames = np.asarray(frames)
        return frames, fps


if __name__ == '__main__':
    train_set = Countix(root=root, split='train')

    # Use index 2093 if using the unaltered dataset (i.e. countix_train.csv, not countix_train_no_missing_ids.csv)
    clip = train_set.get_relevant_clip(1991)
    print(clip.shape)

    # print(train_set.class_dict.items())

    # Visualize single clip
    for frame in clip:
        cv2.imshow('clip', frame)
        k = cv2.waitKey(30) & 0xff
