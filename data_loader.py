import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import ast



class VqaDataset(data.Dataset):
    '''
        Main class use to retrieve our dataset from pickle file.
    '''

    def __init__(self, input_dir, input_vqa, img_feat_vqa, transform=None, phase = 'train'):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True )
        self.img_feat_vqa = np.load(input_dir+'/'+img_feat_vqa, allow_pickle=True )
        
        self.vocab_size = None
        self.phase = phase
        
   

    def __getitem__(self, idx):

        vqa = self.vqa
        img_feat_vqa = self.img_feat_vqa
        
        image_id = vqa['Image_id'].values[idx]
        image_feat = torch.Tensor(img_feat_vqa[image_id])
        

        qst2idc =  vqa['Question'].values[idx]
        sample = { 'image_feature':image_feat ,'question': qst2idc} 
        if (self.phase == 'train') or  (self.phase == 'valid'):
            ans2idc = vqa['labels'].values[idx]
            answer_text = vqa['Answer'].values[idx]
    
            sample['label'] = ans2idc
            sample['answer_text'] = answer_text
        else:
            sample['image_id'] = image_id
            
            
        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, img_feat_train, img_feat_valid, batch_size, num_workers,size=228):
    '''
        Load our dataset with dataloader for the train and valid data
    '''

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            img_feat_vqa=img_feat_train,
            phase = 'train'),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            img_feat_vqa=img_feat_valid,
            phase = 'valid')}
    

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            )
        for phase in ['train','valid']}

    return data_loader


def get_test_loader(input_dir,input_test, img_feat_vqa,batch_size, num_workers,size=228):

   
    
    test_vqa_dataset = VqaDataset(
            input_dir=input_dir,
            input_vqa=input_test,
            img_feat_vqa=img_feat_vqa,
            phase = 'test')
    
    
    data_loader = torch.utils.data.DataLoader(dataset=test_vqa_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
    return data_loader


