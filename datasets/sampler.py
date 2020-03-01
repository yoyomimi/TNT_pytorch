import torch
import torchvision
import torch.utils.data
import random

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.labels = dataset.tracklet_path_list
        
        self.dataset = dict()
        self.balanced_max = 0
        # classify the idx
        for idx in range(0, len(dataset)):
            label = int(self.labels[idx].split()[0].split(',')[8])
            if label not in self.dataset:    
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # oversample to balanced_max
        for label in self.dataset.keys():
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))

        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        group = int(self.balanced_max / len(self.keys))
        while self.indices[self.currentkey] < self.balanced_max - 1 +  group:
            self.indices[self.currentkey] += 1
            label = self.keys[self.currentkey]
            yield random.choice(self.dataset[label])
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
       
    def __len__(self):
        return self.balanced_max*len(self.keys)
