import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import ast


# def make_weights_for_balanced_classes(images, nclasses):                        
#     count = [0] * nclasses                                                      
#     for item in images:                                                         
#         count[item[1]] += 1                                                     
#     weight_per_class = [0.] * nclasses                                      
#     N = float(sum(count))                                                   
#     for i in range(nclasses):                                                   
#         weight_per_class[i] = N/float(count[i])                                 
#     weight = [0] * len(images)                                              
#     for idx, val in enumerate(images):                                          
#         weight[idx] = weight_per_class[val[1]]                                  
#     return weight     





class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, img_feat_vqa, transform=None, phase = 'train'):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True )
        self.img_feat_vqa = np.load(input_dir+'/'+img_feat_vqa, allow_pickle=True )
        # print('here')
        # self.vqa = pd.read_csv(input_dir+'/'+input_vqa)
                    #  , sep=','
                    # , header=True
                    # , index=False
                    # , chunksize=100000
                    # , compression='gzip'
                    # # , encoding='utf-8'
                    # )
        # print('hhhhhey')
        # self.vqa = pd.read_hdf(input_dir+'/'+input_vqa,'mydata')
        #self.class_sample_count = list(self.vqa.labels.value_counts())
        self.vocab_size = None
        self.phase = phase
        # self.transform = transform
   

    def __getitem__(self, idx):

        vqa = self.vqa
        img_feat_vqa = self.img_feat_vqa
        # print('json file',img_feat_vqa)
        # print(type(vqa))
        # print(vqa['Image_feature'])
        
        # transform = self.transform
        #print(vocab_size)
        image_id = vqa['Image_id'].values[idx]
        image_feat = torch.Tensor(img_feat_vqa[image_id])
        # print('feat to tensor',image_feat.shape)
        # print('fresh feat',img_feat_vqa[image_id].shape)
        # image_feat = vqa['Image_feature'].values[idx]
    
        # print(type(image_feat))
        # print(image_feat.shape)
        #image = vqa['Path'].values[idx]
        #image = Image.open(image).convert('RGB')
        #print(image)
        # Convert other data types to torch.Tensor

        qst2idc =  vqa['Question'].values[idx]
        sample = { 'image_feature':image_feat ,'question': qst2idc} #'image': image,
        if (self.phase == 'train') or  (self.phase == 'valid'):
            ans2idc = vqa['labels'].values[idx]
            answer_text = vqa['Answer'].values[idx]
    
            sample['label'] = ans2idc
            sample['answer_text'] = answer_text
        else:
            sample['image_id'] = image_id
            

        # if transform:
        #     sample['image'] = transform(sample['image'])
            
        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, img_feat_train, img_feat_valid, batch_size, num_workers,size=228):

    # transform = {
    #     phase: transforms.Compose([transforms.RandomResizedCrop(size),
    #                                 transforms.ToTensor(),
    #                                transforms.Normalize((0.485, 0.456, 0.406),
    #                                                     (0.229, 0.224, 0.225))]) 
    #     for phase in ['train', 'valid']}

    # dataset_train = datasets.ImageFolder(traindir)                                                                         
                                                                                
    # # For unbalanced dataset we create a weighted sampler                       
    # weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
    # weights = torch.DoubleTensor(weights)                                       
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
    # train_loader = torch.utils.data.DataLoader(dataset_train, 
    #                                             batch_size=batch_size, shuffle = True,                              
    #                                             sampler = sampler, num_workers=workers,
    #                                             pin_memory=True)     

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            img_feat_vqa=img_feat_train,
            #transform=transform['train'],
            phase = 'train'),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            img_feat_vqa=img_feat_valid,
            #transform=transform['valid'],
            phase = 'valid')}
    

    # batch_size = 20
    #class_sample_count =[10, 1, 20, 3, 4] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    # print(class_sample_count)
    # input_vqa ={'train':input_vqa_train, 'valid':input_vqa_valid}
    # sampler ={}
    # for phase in ['train', 'valid']:
    #     vqa = np.load(input_dir+'/'+input_vqa[phase], allow_pickle=True )
    #     class_sample_count = list(vqa.labels.value_counts())
    #     weights = 1 / torch.Tensor(class_sample_count)
    #     sampler[phase] = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size,)

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,#(sampler is None),
            #sampler = sampler[phase],
            num_workers=num_workers,
            #pin_memory=True
            )
        for phase in ['train','valid']}
    # print(iter(data_loader['train']))

    return data_loader


def get_test_loader(input_dir,input_test, img_feat_vqa,batch_size, num_workers,size=228):

    # test_transform = transforms.Compose([transforms.RandomResizedCrop(size),
    #                                 transforms.ToTensor(),
    #                                transforms.Normalize((0.485, 0.456, 0.406),
    #                                                     (0.229, 0.224, 0.225))])


    
    test_vqa_dataset = VqaDataset(
            input_dir=input_dir,
            input_vqa=input_test,
            img_feat_vqa=img_feat_vqa,
            #transform=test_transform,
            phase = 'test')
    
    
    data_loader = torch.utils.data.DataLoader(dataset=test_vqa_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
    return data_loader


