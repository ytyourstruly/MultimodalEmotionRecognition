# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa

from opts import parse_opts
opt = parse_opts()
DEFAULT_SR      = 22050
DEFAULT_N_MFCC  = 40
DEFAULT_N_MELS  = 128
DEFAULT_T_FIXED = 128
DEFAULT_FMAX    = 8000


def _pad_or_truncate(arr: np.ndarray, t_fixed: int) -> np.ndarray:
    t = arr.shape[-1]
    if t == t_fixed:
        return arr
    if t > t_fixed:
        start = (t - t_fixed) // 2
        return arr[..., start:start + t_fixed]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, t_fixed - t)]
    return np.pad(arr, pad_width, mode='constant')


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = arr.mean(axis=-1, keepdims=True)
    std  = arr.std(axis=-1,  keepdims=True) + eps
    return (arr - mean) / std

def get_audio_features(
    y:          np.ndarray,
    sr:         int  = DEFAULT_SR,
    n_mfcc:     int  = DEFAULT_N_MFCC,
    n_mels:     int  = DEFAULT_N_MELS,
    fmax:       int  = DEFAULT_FMAX,
    t_fixed:    int  = DEFAULT_T_FIXED,
    normalise:  bool = True,
    add_deltas: bool = True,
) -> np.ndarray:
    """Returns (C, T_fixed) float32 — C = 255 when add_deltas=True."""
    stft = np.abs(librosa.stft(y))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if add_deltas:
        mfcc_block = np.vstack([
            mfccs,
            librosa.feature.delta(mfccs, order=1),
            librosa.feature.delta(mfccs, order=2),
        ])
    else:
        mfcc_block = mfccs

    log_mel  = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax),
        ref=np.max,
    )
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)

    aligned = [_pad_or_truncate(f, t_fixed)
               for f in [mfcc_block, log_mel, contrast]]
    if normalise:
        aligned = [_zscore(f) for f in aligned]
    result = np.vstack(aligned).astype(np.float32)
    # print(result) 
    return result


def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=opt.audio_channels)
    return mfcc

def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None,
                 fisher_indices_path: str = None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 
        if fisher_indices_path is not None:
                indices = np.load(fisher_indices_path)           # (k,) int64
                # Sort indices so channel order is preserved (spatially coherent)
                self.fisher_indices = np.sort(indices)
                print(f"[RAVDESS] Fisher mask loaded: "
                    f"{len(self.fisher_indices)} / 255 channels kept  "
                    f"({fisher_indices_path})")
        else:
            self.fisher_indices = None                       # use all chann
    def __getitem__(self, index):
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
            
            # features = get_audio_features(y, sr)
            features = get_mfccs(y, sr)
            # if self.fisher_indices is not None:
                # features = features[self.fisher_indices, :]  # (k, T_fixed)          
            audio_features = features 

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
