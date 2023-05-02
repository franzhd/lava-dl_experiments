import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np



class WISDM_Dataset_parser():
    def __init__(self, file_name):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        self.mean = np.mean(x_train, axis=(0,1))
        self.std = np.std(x_train, axis=(0,1))
        print(self.mean.shape)
        print(self.std.shape)
        x_train = x_train - self.mean
        x_train = x_train/self.std

        x_val = x_val - self.mean
        x_val = x_val/self.std

        x_test = x_test - self.mean
        x_test = x_test/self.std

        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[2],x_train.shape[1]))
        x_val = np.reshape(x_val,(x_val.shape[0],x_val.shape[2],x_val.shape[1]))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[2],x_test.shape[1]))
        
        self.train_dataset = (x_train, np.argmax(y_train, axis=-1))
        self.val_dataset = (x_val, np.argmax(y_val, axis=-1))
        self.test_dataset = (x_test, np.argmax(y_test, axis=-1))

    def get_training_set(self, subset=None):
        if subset:
            N = self.train_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None):
        if subset:
            N = self.val_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])
        return self.val_dataset

    def get_test_set(self, subset=None):
        if subset:
            N = self.test_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])
        return self.test_dataset
    
    def de_std(self, data):
        data= data * self.std
        data= data + self.mean

    def load_wisdm2_data(self,file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])


class WISDM_Dataset(Dataset):
     
    def __init__(self, data, target_transform=None):
        x, y = data
        self.x = x
        self.y = y
        self.target_transform = target_transform  

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.target_transform:
            x = self.target_transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.y.size
    
class To_spike(object):
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.thresholds = np.linspace(-1, 1, num_levels)

    def __call__(self, sample):
        
        digitized = np.digitize(sample, self.thresholds) - 1
        out = np.eye(self.num_levels)[digitized]
        return out.reshape(sample.shape[0]*self.num_levels, sample.shape[-1])
        