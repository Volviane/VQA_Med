import os
import argparse
import shutil
import numpy as np
import pandas as pd
import json
import copy
import random
import math
import time
import torch
from data_loader import get_test_loader
import config_1
from models import VqaClassifierModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model1, model2, model3, test_loader, answer_classes_dict):
    since = time.time()
    model1.eval()
    model2.eval()
    model3.eval()
    results = []
    print('Inferencing ...')
    # Iterate over data.
    for batch_idx, batch_sample in enumerate(test_loader):
        image = batch_sample['image_feature'].to(device) #batch_sample['image'].to(device)
        image_id = batch_sample['image_id']#.to(device)
        questions = batch_sample['question']#.to(device)

        output1 = model1(image, questions)
        # output2 = model2(image, questions)
        # output3 = model3(image, questions)
        
        output = output1#+output2)/2

        preds = torch.argmax(output, dim=-1)
        preds = preds.cpu().detach().numpy()

        assert (len(preds) == len(image_id))

        ans_keys = list(answer_classes_dict.keys())
        ans_values = list(answer_classes_dict.values())
    

        for pred, image_name in zip(preds, image_id):
            index_ans = ans_values.index(pred)
            results.append({image_name+'|'+ans_keys[index_ans]})

    df = pd.DataFrame(results)
    df.to_csv('/home/smfogo/submission.csv', index=False)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))





def main():

    input_dir = config_1.input_dir#'/home/smfogo/Med_Visual_question_answering/Exp1_4'#'/content/Med_Visual_question_answering/Exp1_3'#'./'
    input_test = config_1.input_test#'test_dataset_pickle/C1_test_dataset_df.pkl'
    img_feat_test = config_1.img_feat_test#'test_dataset_pickle/test-image-feature.pickle'#.csv.gz'

    with open('answer_classes.json', 'r') as j:
        answer_classes = json.load(j)

    opt = config_1.parse_opt()
        
    
    batch_size = opt.BATCH_SIZE#32 #16
    num_workers = 0
    image_size = opt.IMG_INPUT_SIZE#228
    
    # Create the DataLoader for our dataset
    test_data_loader = get_test_loader(
        input_dir = input_dir , 
        input_test = input_test, 
        img_feat_vqa = img_feat_test,
        batch_size = batch_size, 
        num_workers = num_workers,
        size = image_size )

    model1 = VqaClassifierModel( opt=opt ).to(device)
    model2 = VqaClassifierModel( opt=opt ).to(device)
    model3 = VqaClassifierModel( opt=opt ).to(device)

    saved_dir = config_1.saved_dir#'/home/smfogo/Med_Visual_question_answering/Exp1_4/'#'/content/gdrive/My Drive/vqa/'
    filename1 =saved_dir+'model_state_seed97primetest.tar'
    # filename2 =saved_dir+'model_state_seed97.tar'
    # filename3 =saved_dir+'model_state.tar'
    print("=> loading checkpoint '{}'".format(filename1))
    # print("=> loading checkpoint '{}'".format(filename2)) 
    # print("=> loading checkpoint '{}'".format(filename3)) 

    checkpoint1 = torch.load(filename1)
    # checkpoint2 = torch.load(filename2) 
    # checkpoint3 = torch.load(filename3) 

    start_epoch1 = checkpoint1['epoch'] 

    model1.load_state_dict(checkpoint1['state_dict'])
    # model2.load_state_dict(checkpoint2['state_dict']) 
    # model3.load_state_dict(checkpoint3['state_dict'])  
    #optimizer.load_state_dict(checkpoint['optimizer']) 
    print("=> loaded checkpoint '{}' (epoch {})" .format(filename1, checkpoint1['epoch']))
    inference(model1=model1, model2=model2, model3=model3, test_loader=test_data_loader, answer_classes_dict=answer_classes )
    
if __name__ == '__main__':
    main()
