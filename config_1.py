import argparse
import json
# vqa tools - get from https://github.com/VT-vision-lab/VQA

# location of the data and where to store iamge feature image
path_output_change = '/home/smfogo/Med_Visual_question_answering/Exp1_6'#'/content/Med_Visual_question_answering/Exp1_4'  #'.'
path_change = '/home/smfogo' #'/content' #'.' 
path_output_chd = '/home/smfogo/Med_Visual_question_answering/Exp1_6'  #'/content/Med_Visual_question_answering/Exp1_4'#'.'  

input_dir ='/home/smfogo/Med_Visual_question_answering/Exp1_6'#'/content/Med_Visual_question_answering/Exp1_3/'#'./'
input_vqa_train = 'train_dataset_pickle/train_dataset_df.pkl'
input_vqa_valid ='valid_dataset_pickle/valid_dataset_df.pkl'

img_feat_train = 'train_dataset_pickle/train-image-feature.pickle'
img_feat_valid ='valid_dataset_pickle/valid-image-feature.pickle'

input_dir ='/home/smfogo/Med_Visual_question_answering/Exp1_6'#'/content/Med_Visual_question_answering/Exp1_3'#'./'
input_test = 'test_dataset_pickle/C1_test_dataset_df.pkl'
img_feat_test = 'test_dataset_pickle/test-image-feature.pickle'#.csv.gz'
#location to store the trained model
saved_dir = '/home/smfogo/Med_Visual_question_answering/Exp1_6/'#'/content/gdrive/My Drive/vqa/'



def parse_opt():
    parser = argparse.ArgumentParser()
    with open('/home/smfogo/Med_Visual_question_answering/Exp1_6/answer_classes.json', 'r') as j:
        answer_classes = json.load(j)
    # Data input settings
    parser.add_argument('--SEED', type=int, default=97)
    parser.add_argument('--BATCH_SIZE', type=int, default=64) #32
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=64) 
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=len(answer_classes))
    parser.add_argument('--MAX_QUESTION_LEN', type=int, default=17)
    parser.add_argument('--PRINT_INTERVAL', type=int, default=10)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=50)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=1472) #512 #1984 #2496 #1472
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=1e-4 ) # 1e-4   #0.001=0.5353 #0.002=0.5415 #0.004=0.5481 #0.008= 0.5542, 0.5579
    parser.add_argument('--LAMNDA', type=float, default=0.0001) #1e-3 #0.01 gives 0.5708 with lr_sch_step_size 2, 0.0001 gives 0.5728 same step size
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    parser.add_argument('--DECAY_STEPS', type=int, default=200)
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--BERT_UNIT_NUM', type=int, default=768)
    parser.add_argument('--BERT_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=1) #196 #1 #1
    parser.add_argument('--IMG_INPUT_SIZE', type=int, default=224) # 228
    parser.add_argument('--NUM_EPOCHS', type=int, default=200) #300 
    args = parser.parse_args()
    return args