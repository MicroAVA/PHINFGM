# VHINFGM

## Environment configuration

This project uses Anaconda + Python3.7
Several Python packages required to run the project, in requirements.txt<br>
These Python packages can be installed using Pip or Conda as shown in the following example：<br>
```
pip install -r requirements.txt
```
```
conda install --yes --file requirements.txt
```

***
## Document describing
***There are three folders***<br>

**1.Input folder：**
A folder containing four datasets：<br>
* Dataset Ⅰ:728_129
* Dataset Ⅱ:32_119
* Dataset Ⅲ:312_747
* Dataset Ⅳ:1380_221<br>

Each data set contains virus-host interactions, virus-virus similarity, and host-host similarity

**2.Embedding folder：**
There are also four folders for four data sets, and each folder contains the Node2Vec Embeddings file generated for each fold of the training data


**3.Novel_VHI folder：**
There are also four folders for each of the four data sets to write to the new VHIs

***
***Python文件***
* load_datasets.py-->Read input data, including interaction and similarity
* get_Embedding_FV.py--> Read the inserts generated by Node2vec and get the FV for each virus and host (CV random seed = 22)
* training_functions.py-->Used for several training and processing functions, such as edgeList, Cosine_similarity,..
* pathScores_functions.py-->Calculates and returns all path scores for the path structure
* snf_code.py-->Similarity network fusion function
* GIP.py--> Calculates and returns the GIP similarity
* VHIs_Main_ada.py-->main（3）

## Run
```
python VHIs_Main_ada.py --data 728_129
```
```
python VHIs_Main_ada.py --data 32_119
```
```
python VHIs_Main_ada.py --data 312_747
```
```
python VHIs_Main_ada.py  --data 1380_221
```
