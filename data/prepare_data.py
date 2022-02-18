import os
import math
import librosa
import librosa.feature as lf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data.dataset import random_split
import yaml
from numpy import save, load

config_path = 'D:/git-testy/classifier1/config/'
with open(os.path.join(config_path,'config.yaml')) as c:
    configs = yaml.safe_load(c)

genres_directory = configs["input_path"]
genres = [i for i in os.listdir(genres_directory) if os.path.isdir(os.path.join(genres_directory,i))]


def songsTensors(genres_directory, genres):
    '''
    input: genres directory (string), genres (list).
    Data in a form of ndarray is being saved after preprocessing.
    '''
    songs_tensors = []
    for idx, g in enumerate(genres):
        path = genres_directory + g + '/'
        print(path)
        songs_paths = [os.path.join(path, filename) for filename in os.listdir(path)]
        for s_idx, s in enumerate(songs_paths):
            T = [j for j in range(26)]
            x, sr = librosa.load(s)
            frame_length = math.floor(sr * .25)
            hop_length = math.floor(frame_length / 2)
            if g in s: T[0] = idx
            T[1] = lf.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)[0].mean()
            T[2] = lf.zero_crossing_rate(x).mean()
            T[3] = lf.spectral_centroid(x, sr)[0].mean()
            T[4] = lf.spectral_bandwidth(x + .01, sr)[0].mean()
            T[5] = lf.spectral_rolloff(x + .01, sr)[0].mean()
            mfccs = lf.mfcc(x, sr)
            for k in range(len(mfccs)):
                T[6 + k] = mfccs[k]
            T=np.asarray(T,dtype=object)
            T_stacked = np.hstack(T)
            songs_tensors.append(T_stacked)
            #if s_idx == 2:
                #break
    save(configs["preprocessed_data_path"], np.asarray(songs_tensors,dtype=object))
    return np.asarray(songs_tensors,dtype=object)

class MyDataset(Dataset):
    def __init__(self, songs_tensors):
        self.songs_tensors = songs_tensors
        self.tensors_lengths = np.zeros(len(self.songs_tensors))
        self.tensors_lengths = [len(self.songs_tensors[i]) for i in range(len(self.songs_tensors))]
        self.max_len_tensor = configs["input_size"]
        #print(f'MAX LEN TENSOR {self.max_len_tensor}')            #get max len tensor to determine input_size

    def __len__(self):
        return len(self.songs_tensors)

    def __getitem__(self, idx):
        y_label = torch.tensor(self.songs_tensors[idx][0])
        x_sample = torch.from_numpy(self.songs_tensors[idx][1:])
        max_len_tensor = self.max_len_tensor
        return (y_label, x_sample, max_len_tensor)


def collate_fctn(data):
    labels, samples, maks = zip(*data)
    labels = torch.tensor(np.int64(labels))
    maks = int(maks[0])
    target_samples = torch.zeros(len(data), maks)
    for i in range(len(data)):
        target_samples[i] = torch.cat((samples[i],torch.zeros(maks-len(samples[i]))),0)
    return target_samples.clone().detach(), labels.clone().detach()

# Check if data was already preprocessed in songsTensors function
if os.path.isfile(configs["preprocessed_data_path"]):
    songs_tensors = load(configs["preprocessed_data_path"], allow_pickle=True)
    dataset = MyDataset(songs_tensors)
else:
    dataset = MyDataset(songsTensors(genres_directory, genres))

# Split dataset to train + val and create dataloader
len_train = int(len(dataset)*configs["train_dataset_percent"])
len_val = len(dataset) - len_train
train_dataset, val_dataset = random_split(dataset,[len_train,len_val])
train_loader = DataLoader(train_dataset,batch_size=configs["batch_size"],shuffle=True,collate_fn=collate_fctn)
val_loader = DataLoader(val_dataset,batch_size=configs["batch_size"],shuffle=True,collate_fn=collate_fctn)




