Word/Topic Intrusion Evaluation for Gensim LDA!
=========

- This module is an implementation of a topic model evaluation method from [Reading Tea Leaves: How Humans Interpret Topic Models](http://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf). The method is introduced by people at Princeton, Facebook and University of Maryland in 2009. 
- This module creates necessary datasets and calculate word intrusion precision, topic intrusion log odds and topic intrusion precision.
- This module is built with Python 3.

## How to install
1. Clone this repo and cd into its directory.  
2. run: `pip install -r requirements.txt && python setup.py install` in your shell. This should install all the necessary libraries. 

## Things to prepare
- Prepare gensim LDA model (and the corresponding `.gensim.state file` in the same path), dictionary (generated from Gensim) and documents (see "Data Format" section). 
- In addition to above data and model, you will need to create training datasets (see "Data Format" section). 
- Modify each path for neccesary data and model in `config.py` file. If you know how topic clusters are distributed and want to merge some topics together, you can merge them by following the comments inside the file.

## Two options to evaluate topic intrusion
- There are two ways to evaluate a model; (1) reusable and (2) non-reusable way.
- With (1), you can reuse the training data from evaluators for other models. With (2), the data includes words for each topic and the combination of words is unique to each set. When you want to evaluate your model deeper, (2) will show you how the model picks words for each topics or see if evaluators agree looking at the words for each topic instead of the manually given names for each topic (exercise, recipe etc). In this sense, (2) is a harder evaluation method for models. 
- Select an option in `config.py`. (Follow the comments)

## Model and Data format
Create model and data with the required formats shown below. See the example data in the `data` directory. 
- `model/your_model.gensim`: the default model for this module is a [Gensim LDA model](https://radimrehurek.com/gensim/models/ldamodel.html). You can store the model by `model.save(<filename.gensim>)`. When you store the model, it will also produce `.gensim.state` file and make sure you include the file in the same directory.
- `model/your_dictionary.dict`: you can save the Gensim dictionary by `dictionary.save(<filename.dict>)`. 
- `docs.csv`: each row is a document or message. 
- `wi/word_intrusion_user_set.csv`: wi represents word intrusion. Each row is the word that an evaluater selects as an intruder word.
- `ti/user/x.json`: (when `ti_choice = 'not_reusable_but_debuggable'` in `config.py`) ti represents topic intrusion. Each json file represents the topics for corresponding order number (x) of a document. Inside the json file, the meaning of key and value is `{<topic id>:[<words for the topic>]}`. That's the topic and corresponding words that an evaluator selects as an intruder topic. 
- `ti/user/user_reusable.csv`: (when `ti_choice = 'reusable'` in `config.py`) list of the topics that an evaluator picks as an intruder topic. 
- Note: you can change the path and file name as you like but make sure that you have the right paths in the `config.py`.

## How it works
### Generate datasets
- After creating required data and model and setting up the path, run `python create_datasets.py` and you'll see the output like below if data are successfully created.
```
$ python create_datasets.py 
Calculating...
Merging predetermined topics..
wi data created!
ti data created!
```

### Evaluate the model
- For __word intrusion__, it will calculate the precison, the measurement of topic coherence. This is precision because it measures how well the selected topics include true positives. Example is below.
```
$ python run_word_intrusion.py 
Merging predetermined topics..
Calculating...
Word intrusion precision: 0.75
It took 0.004 mins

```

- For __topic intrusion__, it will calculate two metrics, log odds and precision. Both are to measure that the model’s decomposition of documents into a mixture of topics agrees with human judgments of the document’s content. The beauty of using log odds here is that it is able to capture the level of disagreement between human judgement and the model using assigned probabilities for topics. However, since the log odds is not an intuitve measure because of ranging from negative infinity to zero (higher the better), the module also computes topic intrusion precision. Think of log odds as a model comparison measurement. Example is below.
```
$ python run_topic_intrusion.py 
Merging predetermined topics..
Calculating...
Topic intrusion log odds: -0.391127111865349
Topic intrusion precision: 0.9
It took 0.00493 mins
```
