## Implementation on ERNIE-based text classification


For this project, I extracted 200,000 news headlines from THUCNews. The length of the text is between 20 and 30. There are a total of 10 categories, each with 20,000 items. Data is entered into the model in units of words.

Categories: finance, real estate, stocks, education, technology, society, current affairs, sports, games, entertainment.

The dataset can be downloaded from http://thuctc.thunlp.org/


### Training and Testing Model:

bert
python run.py --model bert

ERNIE
python run.py --model ERNIE
