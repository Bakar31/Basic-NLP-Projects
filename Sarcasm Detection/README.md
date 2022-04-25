The `News Headlines dataset for Sarcasm Detection` is collected from two news website. [The Onion](https://www.theonion.com/) aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from [HuffPost](https://www.huffingtonpost.com/).


## Overview
Total Records -  28,619 
Sarcastic records - 13,635
Non-sarcastic records - 14,984

Each record consists of three attributes:
 - `is_sarcastic`: 1 if the record is sarcastic otherwise 0
 - `headline`: the headline of the news article
 - `article_link`: link to the original news article. Useful in collecting supplementary data

 ### Sarcastic Word Cloud
![alt text](https://github.com/Bakar31/NLP-Projects/blob/master/Sarcasm%20Detection/sarcastic%20word%20cloud.png)
 ### Not Sarcastic Word Cloud
 ![alt text](https://github.com/Bakar31/NLP-Projects/blob/master/Sarcasm%20Detection/non%20sarcastic%20word%20cloud.png)

 ## My approach:
 I made a model with Tensorflow.keras api and used `Conv1D`, `GlobalAveragePooling1D`, `Dropout` layers. Trained the model for 10 epochs and got `94.7` accuracy on train set  `84.31` on test set. *The model didn't overfit*.

 ### Evaluation:
 Accuracy plot:
 ![alt text](https://github.com/Bakar31/NLP-Projects/blob/master/Sarcasm%20Detection/accuracy.png)

 loss plot:
 ![alt text](https://github.com/Bakar31/NLP-Projects/blob/master/Sarcasm%20Detection/loss.png)

 confusion matrix:
![alt text](https://github.com/Bakar31/NLP-Projects/blob/master/Sarcasm%20Detection/confusion%20matrix.png)