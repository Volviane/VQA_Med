# Visual Question Answering in the medical domain

This is our implementation of our model validated on [VQA-Med 2019](https://github.com/abachaa/VQA-Med-2019 )



 ![Figure 1: The MFB+CoAtt Network architecture for VQA Med.](https://github.com/Volviane/VQA_Med/blob/main/imgs/model_architecture.PNG)

 We based our model on [Halim's model](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_85.pdf) and the authors helped us with the script to evaluate our model.
## Requirements
Python 3.8.5, pytorch 1.7.0, torchvision 0.8.1, you have to follow the procedure for the requiment on *training from scratch* part
 
  
## Result
| Models                            	| Modality 	| Plane  	| Organ system 	| Abnormality 	| Overall 	|
|-----------------------------------	|----------	|--------	|--------------	|-------------	|---------	|
| [Team Halim](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_85.pdf) (State-of-the-art)     	| 0.808    	| 0.768  	| 0.736        	| 0.184       	| 0.624   	|
| [Team UMMS](https://www.semanticscholar.org/paper/Deep-Multimodal-Learning-for-Medical-Visual-Shi-Liu/1b0ae121c79437bb122d0cd20d744776445792a4) (fifth at VQA-Med 2019) 	| 0.672    	| 0.760  	| 0.736        	| 0.096       	| 0.566   	|
| Our Model                         	| 0.862    	| 0.752  	| 0.687        	| 0.088       	| 0.597   	|

see details on trainig parameter [here](###)

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