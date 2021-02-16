import argparse
import json


#path on virtual machine 
#****************************To be Change to reproduce ou result*********************************************
path = '/home/smfogo'
#****************************To be Change to reproduce ou result*********************************************

# location of the data and where to store iamge feature image
path_output_chd = path+'/VQA_Med'    

input_vqa_train = 'train_dataset_pickle/train_dataset_df.pkl'
input_vqa_valid ='valid_dataset_pickle/valid_dataset_df.pkl'

img_feat_train = 'train_dataset_pickle/train-image-feature.pickle'
img_feat_valid ='valid_dataset_pickle/valid-image-feature.pickle'

input_test = 'test_dataset_pickle/C1_test_dataset_df.pkl'
img_feat_test = 'test_dataset_pickle/test-image-feature.pickle'




def parse_opt():
    parser = argparse.ArgumentParser()
    with open('/home/smfogo/VQA_Med/answer_classes.json', 'r') as j:
        answer_classes = json.load(j)

    # Data input settings
    parser.add_argument('--SEED', type=int, default=97)
    parser.add_argument('--BATCH_SIZE', type=int, default=64) 
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=64) 
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=len(answer_classes))
    parser.add_argument('--MAX_QUESTION_LEN', type=int, default=17)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=1984)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=1e-4) 
    parser.add_argument('--LAMNDA', type=float, default=0.0001) 
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--BERT_UNIT_NUM', type=int, default=768)
    parser.add_argument('--BERT_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=1) 
    parser.add_argument('--IMG_INPUT_SIZE', type=int, default=224) 
    parser.add_argument('--NUM_EPOCHS', type=int, default=300) 
    args = parser.parse_args()
    return args