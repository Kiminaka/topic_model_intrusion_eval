import evaluation_function as ef
import pandas as pd
import numpy as np
import gensim
from collections import Counter
import time
import re
import config

docs_path = config.docs_path
model_path = config.model_path
dictionary_path = config.dictionary_path
wi_user_filename = config.wi_user_filename
wi_correct_filename = config.wi_correct_filename
wi_merged_set_filename = config.wi_merged_set_filename
num_topics = config.num_topics
ti_user_path = config.ti_user_path
ti_correct_path = config.ti_correct_path
ti_merged_set_path = config.ti_merged_set_path
try:
    print('Merging predetermined topics..')
    merge_ids_list = config.merge_ids_list
except:
    print('No merging topics..')

# Read dataframe in pandas and create a list of docs
start = time.time()
print("Calculating...")
docs = list(pd.read_csv(docs_path, index_col=0, header=None, encoding = "ISO-8859-1").values[1:])
# Read gensim model and dictionary
ldamodel = gensim.models.LdaModel.load(model_path)
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
topics = ldamodel.show_topics(num_topics=num_topics, num_words=100, formatted=False)

user_wi_list = pd.read_csv(wi_user_filename).drop('Unnamed: 0', axis=1).values
correct_wi_list = pd.read_csv(wi_correct_filename).drop('Unnamed: 0', axis=1).values
# calculation
precision = ef.calc_wi_precision(correct_wi_list, user_wi_list)

end = time.time()

if __name__ == "__main__":
    print("Word intrusion precision: {}".format(precision))
    print("It took {0:.3g} mins".format((end-start)/60))
