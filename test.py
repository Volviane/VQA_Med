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


def inference(model, test_loader, answer_classes_dict, path_change):
    since = time.time()
    model.eval()
    results = []
    print('Inferencing ...')
    # Iterate over data.
    for batch_idx, batch_sample in enumerate(test_loader):
        image = batch_sample['image_feature'].to(device) 
        image_id = batch_sample['image_id']
        questions = batch_sample['question']

        output = model(image, questions)
        preds = torch.argmax(output, dim=-1)
        preds = preds.cpu().detach().numpy()

        assert (len(preds) == len(image_id))

        ans_keys = list(answer_classes_dict.keys())
        ans_values = list(answer_classes_dict.values())
    

        for pred, image_name in zip(preds, image_id):
            index_ans = ans_values.index(pred)
            results.append({image_name+'|'+ans_keys[index_ans]})

    df = pd.DataFrame(results)
    df.to_csv(path_change+'/submission.csv', index=False)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))





def main():

   
    path_output_change =config_1.path_output_change
    path_change = config_1.path_change

    input_dir = config_1.input_dir
    input_test = config_1.input_test
    img_feat_test = config_1.img_feat_test

    with open(path_output_change+'/answer_classes.json', 'r') as j:
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

    model = VqaClassifierModel( opt=opt ).to(device)
    saved_dir = config_1.saved_dir
    filename =saved_dir+'model_state_seed_97.tar'
    print("=> loading checkpoint '{}'".format(filename)) 
    checkpoint = torch.load(filename) 
    start_epoch = checkpoint['epoch'] 
    model.load_state_dict(checkpoint['state_dict']) 
  
    print("=> loaded checkpoint '{}' (epoch {})" .format(filename, checkpoint['epoch']))
    inference(model=model, test_loader=test_data_loader, answer_classes_dict=answer_classes , path_change= path_change)
    
if __name__ == '__main__':
    main()
