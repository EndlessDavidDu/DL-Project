## Implementation on ERNIE-based text classification

For this project, I extracted 200,000 news headlines from THUCNews. The length of the text is between 20 and 30. There are a total of 10 categories, each with 20,000 items. Data is entered into the model in units of words.

Categories: finance, real estate, stocks, education, technology, society, current affairs, sports, games, entertainment.

The dataset can be downloaded from http://thuctc.thunlp.org/


This project is mainly divided into two parts. One part is the Chinese news classification model based on Word2vec word vector, and the other part is the classification model based on Ernie and Bert. These two sections are placed in two folders. In each section, the folder "model" defines the relevant model and the configuration information for each model. The results of the training of the model are saved in "save" and "save_information".
In addition, the LoggerClass.py file is primarily used for logging. Utils.py focuses on some common methods. run.py is the entry point to my program, and by calling it, the model is trained. The specific invocation method is as follows:
python run.py --model ERNIE

In addition, the pre-training language models are placed in their own folders. For example, the Bert model is in the bert_pretain directory and the Ernie model is in the ERNIE_pretrain directory.
