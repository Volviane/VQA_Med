#import standard package
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import re
import collections
import pickle
import os
from tqdm import tqdm
#from sklearn.preprocessing import LabelEncoder
import json
import config_1

import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as f
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Extract image feature
class VGG16(nn.Module):
    def __init__(self):
        '''
            We remove all the fully-connected layers in the VGG16 network and convolution outputs of last feature scales
             before the Adapdative average pooling is use to represent a 512x14x14 tensor to represent the image
         --------------------------------------------------------------------------------------------------------------------------------
             We remove all the fully-connected layers in the VGG16 network and the convolution outputs of different feature scales
                are concatenated after global average pooling to form a 1984-dimensional vector to represent the image
        '''
        super(VGG16,self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)	
        # print(vgg_model)

        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9])
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])
        self.Conv6 = nn.Sequential(*list(vgg_model.features.children())[30:36])

        self.avgpool = nn.Sequential(list(vgg_model.children())[1])
    
    def forward(self,image):

        with torch.no_grad():
            out1 = self.Conv1(image)
            out2 = self.Conv2(out1)
            out3 = self.Conv3(out2)
            out4 = self.Conv4(out3)          
            out5 = self.Conv5(out4)          # [N, 512, 14, 14]
            out6 = self.Conv6(out5) 

            out7 = self.avgpool(out6)

        #global average pooling
        out1 = out1.mean([2,3],keepdim=True)
        out2 = out2.mean([2,3],keepdim=True)
        out3 = out3.mean([2,3],keepdim=True)
        out4 = out4.mean([2,3],keepdim=True)
        out5 = out5.mean([2,3],keepdim=True)
        out6 = out6.mean([2,3],keepdim=True)
        out7 = out7.mean([2,3],keepdim=True)

        #l2-normalized feature vector
        # out1_norm = out1.norm(p=2, dim=1, keepdim=True).detach() 
        # out1 = out1.div(out1_norm)               # l2-normalized feature vector

        # out2_norm = out2.norm(p=2, dim=1, keepdim=True).detach() 
        # out2 = out2.div(out2_norm)               # l2-normalized feature vector

        # out3_norm = out3.norm(p=2, dim=1, keepdim=True).detach() 
        # out3 = out3.div(out3_norm)               # l2-normalized feature vector

        # out4_norm = out4.norm(p=2, dim=1, keepdim=True).detach() 
        # out4 = out4.div(out4_norm)               # l2-normalized feature vector

        # out5_norm = out5.norm(p=2, dim=1, keepdim=True).detach() 
        # out5 = out5.div(out5_norm)               # l2-normalized feature vector

        # out6_norm = out6.norm(p=2, dim=1, keepdim=True).detach() 
        # out6 = out6.div(out6_norm)               # l2-normalized feature vector

        # out7_norm = out7.norm(p=2, dim=1, keepdim=True).detach() 
        # out7 = out7.div(out7_norm)               # l2-normalized feature vector



        # out1 = f.normalize(out1,dim=0,p=2)
        # out2 = f.normalize(out2,dim=0,p=2)
        # out3 = f.normalize(out3,dim=0,p=2)
       
        # # print(out1.shape, out2.shape, out3.shape, out4.shape, out5.shape)
        concat_features = torch.cat([out1, out2, out3, out4,out5], 1) #, out5, out6,out7

        # #l2-normalized feature vector
        l2_norm = concat_features.norm(p=2, dim=1, keepdim=True).detach() 
        concat_features = concat_features.div(l2_norm)               # l2-normalized feature vector
       
        


        #l2-normalized feature vector
        # l2_norm = out6.norm(p=2, dim=1, keepdim=True).detach() 
        # concat_features = out6.div(l2_norm)               # l2-normalized feature vector

        # batch_size = out6.shape[0]
        # embedding_dim_size = out6.shape[1]
        # image_feature = out6.view(batch_size, embedding_dim_size, -1).squeeze(0) # [N, 512, 196]

        batch_size = concat_features.shape[0]
        embedding_dim_size = concat_features.shape[1]
        image_feature = concat_features.view(batch_size, embedding_dim_size, -1).squeeze(0) # [N, 1984, 1]


        return image_feature


def parse_sentence(s):
    #s = s.replace('.', '')
    #s = s.replace(',', '')
    #s = s.replace('"', '')
    #s = s.replace("'", '')
    #s = s.replace("?", '')
    s = s.replace(" - ", "-")
    s = s.lower()
    s = re.sub("\s\s+", " ", s)
    #s = s.split(' ')
    return s


#read a txt file for each category and structure it in a dataframe
def get_category_file_train(category_name, category_path ,images_path, vgg16_model, transform=None):#group='train'
    
    data = []
    
    dict_data = { 'Image_id':[],
                  #'Image_feature': [],
                'Question':[],
                'Answer':[],
                #'Group':[],
                #'Path':[]
              }
    image_feat  = { }
    with open(category_path) as f:
        lines = f.readlines()
    
    for element in lines:
        pd_element = element.split('|')
        dict_data['Image_id'].append(pd_element[0])
        dict_data['Question'].append(re.sub(r'\s+', ' ',pd_element[1]).strip())
        dict_data['Answer'].append(pd_element[2].strip("\n")) #parse_sentence()
        #dict_data['Group'].append(group)
        #dict_data['Path'].append(images_path+pd_element[0]+'.jpg')
        image = Image.open(images_path+pd_element[0]+'.jpg').convert('RGB')
        if transform:
            image = transform(image)
        image_feature = vgg16_model(image[None,...].to(device))
        # print(image_feature.shape)
        # print(image_feature)
        #dict_data['Image_feature'].append(image_feature)
        image_feat[pd_element[0]] = image_feature.cpu().numpy()
        

        
    df_data = pd.DataFrame(dict_data, columns = ['Image_id',
                                                 #'Image_feature',
                                                 'Question',
                                                 'Answer',
                                                 #'Group', 
                                                 #'Path'
                                                 ])


    answer_freq = count_answer_freq(df_data)
    # for a in answer_freq:
    #     print(a)
    # print('number of answer_freq', len(answer_freq))
    classes = get_most_frequent_classes(answer_freq, threshold=5)
    # print('number of classes', len(classes))
    df_data['labels'] = df_data['Answer'].apply(lambda x : classes[x] if x in classes else classes['UNKNOWN'])
    # print(classes)
    count =0
    
    
    return df_data, classes,image_feat

    #read a txt file for each category and structure it in a dataframe
def get_category_file_valid(category_name, category_path ,images_path, classes, vgg16_model, transform=None):#group='valid'
    
    data = []
    
    dict_data = { 'Image_id':[],
                'Question':[],
                'Answer':[],
                #'Group':[],
                #'Path':[]
              }
    image_feat = { }
    with open(category_path) as f:
        lines = f.readlines()
    
    for element in lines:
        pd_element = element.split('|')
        dict_data['Image_id'].append(pd_element[0])
        dict_data['Question'].append(re.sub(r'\s+', ' ',pd_element[1]).strip())
        dict_data['Answer'].append(pd_element[2].strip("\n")) #parse_sentence()
        #dict_data['Group'].append(group)
        #dict_data['Path'].append(images_path+pd_element[0]+'.jpg')
        image = Image.open(images_path+pd_element[0]+'.jpg').convert('RGB')
        if transform:
            image = transform(image)
        image_feature = vgg16_model(image[None,...].to(device))
        #dict_data['Image_feature'].append(image_feature)
        image_feat[pd_element[0]] = image_feature.cpu().numpy()
        
    df_data = pd.DataFrame(dict_data, columns = ['Image_id', 
                                                 'Question',
                                                 'Answer',
                                                 #'Group', 
                                                 #'Path'
                                                 ])
    # knw_idx = len(classes)
    
    # print(classes)
    df_data['labels'] = df_data['Answer'].apply(lambda x : classes[x] if x in classes else classes['UNKNOWN'])

    
    return df_data ,image_feat


def get_test_file(category_name, category_path ,images_path,vgg16_model, transform=None,group='test'):
    
    data = []
    
    dict_data = { 'Image_id':[],
                'Question':[],
                'Group':[],
                #'Path':[]
              }

    image_feat = { }
    with open(category_path) as f:
        lines = f.readlines()
    
    for element in lines:
        pd_element = element.split('|')
        dict_data['Image_id'].append(pd_element[0])
        dict_data['Question'].append(re.sub(r'\s+', ' ',pd_element[1]).strip("\n")) #parse_sentence(pd_element[1].strip("\n"))
        dict_data['Group'].append(group)
        #dict_data['Path'].append(images_path+pd_element[0]+'.jpg')
        image = Image.open(images_path+pd_element[0]+'.jpg').convert('RGB')
        if transform:
            image = transform(image)
        image_feature = vgg16_model(image[None,...].to(device))
        #dict_data['Image_feature'].append(image_feature)
        image_feat[pd_element[0]] = image_feature.cpu().numpy()
        
    df_data = pd.DataFrame(dict_data, columns = ['Image_id',
                                                 'Question',
                                                 'Group', 
                                                 #'Path'
                                                 ])
    return df_data,image_feat

def count_answer_freq(df_data):
    '''
    count the frequence of each unique answer on the dataset
    '''
    all_answers = df_data['Answer'].values
    answer_freq_dict = defaultdict(int)
    for answer in all_answers:
        answer_freq_dict[answer] += 1
    answer_freq_dict_sort = dict(sorted(answer_freq_dict.items(), key=lambda x: x[1], reverse=True))
    # print(type(answer_freq_dict_sort))

    return answer_freq_dict_sort

def plot(answer_freq_dict):
    keys = answer_freq_dict.keys()
    values = answer_freq_dict.values()
    plt.bar(keys, values)
    plt.title('Frequences of answer on the dataset')
    plt.xlabel('answer')
    #plt.xlim([0, 1552])
    plt.ylim([1, 600])
    plt.ylabel('frquence')
    plt.savefig('answer_frequence.png')
    #plt.show()

def get_most_frequent_classes(answer_freq_dict, threshold=1):
    final_classes = defaultdict(int)
    index = 0

    for answer,ans_freq in answer_freq_dict.items():

        if ans_freq >= threshold:
            final_classes[answer]= index         
        else:
            final_classes['UNKNOWN'] = index
            break
        index += 1  
    final_classes['UNKNOWN'] = index
    with open('answer_classes.json', 'w') as fp:
        json.dump(final_classes, fp)
    return final_classes


        



#save preprocessed training data frame
def save_preprocesse_data(dataset_df, pickle_name, image_feat, name= 'train'):
    path_output_chd = config_1.path_output_chd#'/home/smfogo/Med_Visual_question_answering/Exp1_4'  #'/content/Med_Visual_question_answering/Exp1_4'#'.'  
    if name == 'train':
        
        dataset_df.to_pickle(path_output_chd+'/train_dataset_pickle/'+pickle_name+'.pkl')
        with open(path_output_chd+'/train_dataset_pickle/train-image-feature.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(image_feat, f, pickle.HIGHEST_PROTOCOL)
        
    elif name == 'valid':
        
        dataset_df.to_pickle(path_output_chd+'/valid_dataset_pickle/'+pickle_name+'.pkl')

        with open(path_output_chd+'/valid_dataset_pickle/valid-image-feature.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(image_feat, f, pickle.HIGHEST_PROTOCOL)
        
    else: 
       
        dataset_df.to_pickle(path_output_chd+'/test_dataset_pickle/'+pickle_name+'.pkl')
        with open(path_output_chd+'/test_dataset_pickle/test-image-feature.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(image_feat, f, pickle.HIGHEST_PROTOCOL)




def main():

    category_names = {'C1': 'Modality',
                    'C2': 'Plane',
                    'C3': 'Organ',
                    'C4': 'Abnormality',
                    'All': 'All dataset',}
    
    opt = config_1.parse_opt()
    path_output_change =config_1.path_output_change#'/home/smfogo/Med_Visual_question_answering/Exp1_4'#'/content/Med_Visual_question_answering/Exp1_4'  #'.'
    path_change = config_1.path_change#'/home/smfogo' #'/content' #'.' 
    #set the seed
    seed_value = opt.SEED
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
 
    # le = LabelEncoder()

    train_path = path_change+'/VQA_Med_2019_Dataset/Train/ImageClef-2019-VQA-Med-Training/'#QAPairsByCategory/'
    train_images_path = path_change+'/VQA_Med_2019_Dataset/Train/ImageClef-2019-VQA-Med-Training/Train_images/'

    valid_path = path_change+'/VQA_Med_2019_Dataset/Valid/ImageClef-2019-VQA-Med-Validation/'#QAPairsByCategory/'
    valid_images_path = path_change+'/VQA_Med_2019_Dataset/Valid/ImageClef-2019-VQA-Med-Validation/Val_images/'

    test_path = path_change+'/VQA_Med_2019_Dataset/Test/VQAMed2019Test/' 
    test_images_path = path_change+'/VQA_Med_2019_Dataset/Test/VQAMed2019Test/VQAMed2019_Test_Images/'
    size = opt.IMG_INPUT_SIZE#228
    transform = {
        phase: transforms.Compose([transforms.RandomResizedCrop(size), #transforms.CenterCrop(size),
                                    #transforms.ColorJitter(),
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
    
        for phase in ['train', 'valid']}
    
    test_transform = transforms.Compose([transforms.RandomResizedCrop(size), #transforms.CenterCrop(size),
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
    
    vgg16_model = VGG16().to(device)

    train_dataset_df, classes, image_feat_train = get_category_file_train(category_name=category_names['All'],#C1 
                                                        category_path=train_path+'All_QA_Pairs_train.txt', #'All_QA_Pairs_train.txt',#'C1_Modality_train.txt', # 'C2_Plane_train.txt', # 'C3_Organ_train.txt',# 'C4_Abnormality_train.txt',
                                                        images_path=train_images_path,
                                                        vgg16_model=vgg16_model,
                                                        transform=transform['train'])

    valid_dataset_df, image_feat_valid = get_category_file_valid(category_name=category_names['All'],#C1
                                                    category_path=valid_path+ 'All_QA_Pairs_val.txt', #'C3_Organ_val.txt',#'All_QA_Pairs_val.txt',#'C1_Modality_val.txt', # 'C2_Plane_val.txt', # 'C3_Organ_val.txt',# 'C4_Abnormality_val.txt',
                                                    images_path=valid_images_path,
                                                    classes= classes,
                                                    vgg16_model=vgg16_model,
                                                    transform=transform['valid'])

    test_dataset_df, image_feat_test = get_test_file(category_name=category_names['All'], 
                                                  category_path=test_path+'VQAMed2019_Test_Questions.txt'  ,
                                                  images_path=test_images_path,
                                                  vgg16_model=vgg16_model,
                                                  transform=test_transform,
                                                  group='test')
    C1_test_dataset_df= test_dataset_df[:] #[:125] [125:250] [250:375] [375:]
    
  
    answer_freq = count_answer_freq(train_dataset_df)
 

    

    
    ####################################################SAVE AND LOAD DATA FRAME########################################################
    save_preprocesse_data(train_dataset_df, 'train_dataset_df', image_feat_train,'train')
    save_preprocesse_data(valid_dataset_df, 'valid_dataset_df', image_feat_valid,'valid')
    save_preprocesse_data(C1_test_dataset_df, 'C1_test_dataset_df',image_feat_test, 'test')
   
    print('========>done=======>>>>>>>>>>>>>>')

if __name__ == '__main__':
    main()