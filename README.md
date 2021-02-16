# Visual Question Answering in the medical domain

This is our implementation of our model validated on [VQA-Med 2019](https://github.com/abachaa/VQA-Med-2019 )



 ![Figure 1: The MFB+CoAtt Network architecture for VQA Med.](https://github.com/Volviane/VQA_Med/blob/main/imgs/model_architecture.PNG)

To reproduce our result:


1. Download and unzip the dataset at  https://github.com/abachaa/VQA-Med-2019 ;
2. clone this repo using git clone  git clone https://github.com/Volviane/VQA_Med.git
Please make sure that your unzip data is store in the same diretory as the  VQA_Med project you just clone

Make sure you modify config.py file to change the path, (e.g. in our case here, '/home/smfogo/' is the folder where we stored the data and the  VQA_Med  project )

3. go into the project directory using cd VQA_Med
4. run the following command to install all the dependencies you need to run this project :
    - *conda env create -f vqamed.yml*  if you're using conda 
    or
    - *pip install -r requirements.txt* if you're using pip

    then end run *conda activate vqamed* if you are unsing conda

5. run *python dataset_helper.py* to generate the features of your dataset (this can take few minutes)
6. Finally run python train.py to train your model 