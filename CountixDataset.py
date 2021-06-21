import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import pandas as pd

root = os.path.join('D:', 'Countix')

class Countix(Dataset):

  def __init__(self, root, split, clip_len=16):
    self.root = root
    self.split = split
    self.clip_len = clip_len
    self.annotations = self.read_split()

  def __len__(self):
    return len(self.annotations)

  # Trims clip to start and end points specified in dataset
  def get_relevant_clip(self, index):
    frames, fps = self.read_video(index)

    print('FPS = ' + str(fps))
    print(frames.shape)

    start_index = int(fps * self.annotations['repetition_start'][index])
    end_index = int(fps * self.annotations['repetition_end'][index])
    return frames[start_index : end_index]

  def read_split(self):
    return pd.read_csv(os.path.join(self.root, "CSVs", "Countix_" + self.split + ".csv"), sep=",")

  def read_video(self, index, width=224, height=224):
    """Read video from file."""
    # print('Video ID = ' + self.annotations['video_id'][index])

    cap = cv2.VideoCapture(os.path.join(self.root, self.split, self.annotations['video_id'][index] + '.mp4'))
    if not cap.isOpened():
        cap = cv2.VideoCapture(os.path.join(self.root, self.split, self.annotations['video_id'][index] + '.webm'))
    assert cap.isOpened(), 'Unable to open video in path ' + os.path.join(self.root, self.split, self.annotations['video_id'][index])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    if cap.isOpened():
      while True:
        success, frame_bgr = cap.read()
        if not success:
          break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (width, height))
        frames.append(frame_rgb)
    frames = np.asarray(frames)
    return frames, fps

if __name__ == '__main__':
  train_set = Countix(root=root, split='train')

  print('FPS = ' + str(fps))
  print(frames.shape)