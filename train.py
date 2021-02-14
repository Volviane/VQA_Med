import os
import argparse
import shutil
import numpy as np
import json
import copy
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import get_loader

from models import VqaClassifierModel
import config_1

import matplotlib.pyplot as plt
import nltk
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

#from sklearn.preprocessing import LabelEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt')
opt = config_1.parse_opt()

seed_value = opt.SEED
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def train_model():
    since = time.time()

    
    best_acc1 = 0.0
    best_acc5 = 0.0
    
    #best_acc_val = 0.0
    best_epoch = 0
    list_train_loss_per_epoch = []
    list_valid_loss_per_epoch = []

    list_train_acc1_per_epoch = []
    list_valid_acc1_per_epoch = []

   

    model = VqaClassifierModel( opt=opt ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.INIT_LERARNING_RATE, weight_decay=opt.LAMNDA) 

    input_dir = config_1.input_dir 
    input_vqa_train =config_1.input_vqa_train 
    input_vqa_valid =config_1.input_vqa_valid

    img_feat_train = config_1.img_feat_train
    img_feat_valid =config_1.img_feat_valid

    saved_dir = config_1.saved_dir
  
    
    num_epochs = opt.NUM_EPOCHS
    image_size = opt.IMG_INPUT_SIZE
    num_workers = 0
    batch_size = opt.BATCH_SIZE


    # Create the DataLoader for our dataset

    data_loader = get_loader(
        input_dir = input_dir , 
        input_vqa_train = input_vqa_train, 
        input_vqa_valid = input_vqa_valid,
        img_feat_train = img_feat_train, 
        img_feat_valid = img_feat_valid,
            batch_size = batch_size, 
            num_workers = num_workers,
            size = image_size )

    
    for epoch in range(opt.NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            #accuracy = 0
            top1_acc = 0
            top5_acc = 0
            acc_test_f = 0

            bleu = 0
            batch_step_size = len(data_loader[phase].dataset) / batch_size
        
            # Iterate over data.
            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                
                #image = batch_sample['image'].to(device)
                image = batch_sample['image_feature'].to(device)
                questions = batch_sample['question']#.to(device)
                labels = batch_sample['label'].to(device)
                label_answer_text = batch_sample['answer_text']#.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output= model(image, questions) 
                    
                  
                    _, preds = torch.max(output, 1)
                    
                   
                    loss = criterion(output, labels)
                   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                        
                #exact match score
                acc1, acc5 = accuracy(output.data, labels.data, topk=(1, 5))
                top1_acc += acc1
                top5_acc += acc5
                #bleu score
                b = get_bleu_score(preds, label_answer_text)
                bleu += b
                

                if batch_idx % 10 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step[{:04d}/{:04d}], Loss: {:.4f}, Top 1 Acc: {:.4f}, Top 5 Acc: {:.4f}, Bleu: {:.4f}'.format(phase.upper(), epoch+1, num_epochs, batch_idx, int(batch_step_size), loss.item(), acc1, acc5, b))#Acc: {:.4f},Bleu: {:.4f},acc, b


            epoch_loss = running_loss/batch_step_size
            epoch_acc1 = top1_acc/batch_step_size
            epoch_acc5 = top5_acc/batch_step_size
           
            epoch_blue = bleu/batch_step_size
            
            #save the loss and accuracy for train and valid
            if phase =='train':
              
                list_train_loss_per_epoch.append(epoch_loss)
                list_train_acc1_per_epoch.append(epoch_acc1)
               
            else:
                
                list_valid_loss_per_epoch.append(epoch_loss)
                list_valid_acc1_per_epoch.append(epoch_acc1)
                

            print('{} Loss: {:.4f} Top 1 Acc: {:.4f} Top 5 Acc: {:.4f} Bleu: {:.4f}'.format(
                phase, epoch_loss, epoch_acc1,epoch_acc5, epoch_blue))
            

            # deep copy the model
            if phase == 'valid' and epoch_acc1 > best_acc1: #or epoch_acc5 > best_acc5 ):
                best_acc1 = epoch_acc1
                best_acc5 = epoch_acc5
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

               

    history_loss = {'train':list_train_loss_per_epoch, 'valid':list_valid_loss_per_epoch}
    history_acc1 = {'train':list_train_acc1_per_epoch, 'valid':list_valid_acc1_per_epoch}
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Top 1 Acc: {:4f}, Top 5 Acc: {:4f}'.format(best_acc1,best_acc5))

    #plot the loss and accuracy for train and valid
    make_plot(history_loss, num_epochs, type_plot='loss')
    make_plot(history_acc1, num_epochs, type_plot='acc1')
   

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.load_state_dict(best_model_wts)
    state = {'epoch': best_epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
                'loss':epoch_loss,'valid_accuracy': best_acc1}
    
    full_model_path =saved_dir+'model_state_seed_97.tar'
   
    torch.save(state, full_model_path)
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
   
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
   

    
    if target.dim() == 2: # multians option
        _, target = torch.max(target, 1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size))
   
    return res
        
 
def get_bleu_score(predicted, true_ans_text):
    path_output_change = config_1.path_output_change
    with open(path_output_change+'/answer_classes.json', 'r') as j:
        answer_classes_dict = json.load(j)
    score = 0.0
    assert (len(predicted) == len(true_ans_text))
    ans_keys = list(answer_classes_dict.keys())
    ans_values = list(answer_classes_dict.values())
    

    for pred, true_ans in zip(predicted, true_ans_text):
        index_ans = ans_values.index(pred)
        
        score += sentence_bleu([true_ans.split(' ')], ans_keys[index_ans].split(' '), smoothing_function=bleu_score.SmoothingFunction().method2)

    return score/len(true_ans_text)


def load_checkpoint(model, optimizer, filename=None): 
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states. 
    start_epoch = 0 
    if os.path.isfile(filename): 
        print("=> loading checkpoint '{}'".format(filename)) 
        checkpoint = torch.load(filename) 
        start_epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        print("=> loaded checkpoint '{}' (epoch {})" .format(filename,
                                                            checkpoint['epoch'])) 
    else: print("=> no checkpoint found at '{}'".format(filename)) 
    return model, optimizer, start_epoch
    


def make_plot(history, epoch_max, type_plot='loss'):
    train = history['train']
    valid = history['valid']
    fig, ax = plt.subplots()
    epochs = range(epoch_max)
    
    if type_plot=='loss':
        plt.plot(epochs, train, '-r', lw=2, label='Training loss')
        plt.plot(epochs, valid, '-b',lw=2, label='validation loss')
        plt.legend(borderaxespad=0.)
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss.png')
        
    elif type_plot == 'acc1':
    
        plt.plot(epochs, train, '-r', lw = 2, label='Training Top 1 Accuracy')
        plt.plot(epochs, valid, '-b', lw = 2, label='validation Top 1 Accuracy')
        plt.legend(borderaxespad=0.)
        plt.title('Training and Validation Top 1 Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Top 1 Accuracy')
        plt.savefig('acc1.png')

    elif type_plot == 'acc5':
    
        plt.plot(epochs, train, '-r', lw = 2, label='Training Top 5 Accuracy')
        plt.plot(epochs, valid, '-b', lw = 2, label='validation Top 5 Accuracy')
        plt.legend(borderaxespad=0.)
        plt.title('Training and Validation Top 5 Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Top 5 Accuracy')
        plt.savefig('acc5.png')
    else:
        plt.plot(epochs, train, '-r', lw = 2, label='Training blue')
        plt.plot(epochs, valid, '-b', lw = 2, label='validation blue')
        plt.legend(borderaxespad=0.)
        plt.title('Training and Validation blue')
        plt.xlabel('Epochs')
        plt.ylabel('Blue')
        plt.savefig('blue.png')

    
    
    plt.show()





def main():
    train_model()

    

if __name__ == '__main__':
    main()
