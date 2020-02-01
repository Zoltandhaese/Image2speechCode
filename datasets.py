import torch
from torch.utils.data import Dataset
import os
import pickle
from torch.utils.data.dataloader import default_collate
import numpy as np

def pad_collate_train(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')    
    for elem in batch:        
        imgs, caps, caplen = elem
        max_input_len = max_input_len if max_input_len > caplen else caplen      

    for i, elem in enumerate(batch):
        imgs, caps, caplen = elem
        input_dim = caps.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float)
        feature[:caps.shape[0], :caps.shape[1]] = caps       
        
        batch[i] = (imgs, feature, caplen)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    # batch.sort(key=lambda x: x[-1], reverse=True)
    
    return default_collate(batch)


def pad_collate_test(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')    
    for elem in batch:        
        imgs, caps, caplen, allcaps = elem
        for allcap in allcaps:
            cpalen_a = allcap.shape[0]
            max_input_len = max_input_len if max_input_len > cpalen_a else cpalen_a      

    for i, elem in enumerate(batch):
        imgs, caps, caplen, allcaps = elem
        input_dim = caps.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float)
        feature[:caps.shape[0], :caps.shape[1]] = caps   
        all_features = []
        for allcap in allcaps:
            all_feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            all_feature[:allcap.shape[0], :allcap.shape[1]] = allcap 
            all_features.append(all_feature)
        all_features = np.array(all_features)
        
        batch[i] = (imgs, feature, caplen, all_features)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-2], reverse=True)
    
    return default_collate(batch)




class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, path, split='TRAIN', transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.cpi=5
        self.split = split
        if split == 'TRAIN':
            image_feature_path = os.path.join(path,'train5_new.npz')
            embedding_path = os.path.join(path,'embedded_caps.pickle')
        elif split == 'VAL':
            image_feature_path = os.path.join(path,'test5_new.npz')
            embedding_path = os.path.join(path,'embedded_caps_text.pickle')
        self.img_features = np.load(image_feature_path)
        with open(embedding_path,'rb') as f:
            self.captions = pickle.load(f)

        self.filenames = self.img_features.files
        self.file_idxs = []
        for i in range(len(self.captions)):
            if len(self.captions[i])>2:
                self.file_idxs.append(i)
        self.file_idxs = self.file_idxs
        self.dataset_size = len(self.file_idxs)

    def __getitem__(self, i):
        if self.split == 'TRAIN':
            indx = self.file_idxs[i]
        else:
            indx = i
        file_name = self.filenames[indx]
        img = self.img_features[file_name]
        caption =np.array(self.captions[indx])
        caption = caption[:,np.newaxis]
        caplen = caption.shape[0]

        return img, caption, caplen
        
        """
        if self.split == 'TRAIN':
            return img, caption, caplen
        
        
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]
            all_caps = []
            for all_cap in all_captions:
                all_cap = np.array(all_cap)[:,np.newaxis]
                all_caps.append(all_cap)

            return img, caption, caplen, all_caps 
        """

    def __len__(self):
        return self.dataset_size
