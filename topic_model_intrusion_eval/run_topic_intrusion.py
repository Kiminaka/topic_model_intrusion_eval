import evaluation_function as ef
import pandas as pd
import numpy as np
import gensim
from collections import Counter
import time
import re
import config
import json


docs_path = config.docs_path
model_path = config.model_path
dictionary_path = config.dictionary_path
wi_user_filename = config.wi_user_filename
wi_correct_filename = config.wi_correct_filename
wi_merged_set_filename = config.wi_merged_set_filename
num_topics = config.num_topics

ti_user_path = config.ti_user_path
ti_user_path_reusable = config.ti_user_path_reusable

ti_correct_path = config.ti_correct_path
ti_correct_path_reusable = config.ti_correct_path_reusable

ti_merged_set_path = config.ti_merged_set_path
ti_merged_set_path_reusable = config.ti_merged_set_path_reusable

ti_topic_dist_dict = config.ti_topic_dist_dict

ti_choice = config.ti_choice

try:
    topic_names = config.topic_names
except:
    print('Using non-reusable option..')

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

if ti_choice == 'reusable':
    user_list = list(pd.read_csv(ti_user_path_reusable).drop('Unnamed: 0', axis=1).values)
    ti_list = list(pd.read_csv(ti_correct_path_reusable, ).drop('Unnamed: 0', axis=1).values)

    # convert topic names to topic ids
    user_list = [ef.topic_converter(x, reverse=True) for x in user_list]
    ti_list = [ef.topic_converter(x, reverse=True) for x in ti_list]

    dist_list = []
    for i, _ in enumerate(docs):
        with open(ti_topic_dist_dict.format(i), 'r') as fp:
            dist_list.append(json.load(fp))

if ti_choice == 'not_reusable_but_debuggable':
    user_list = []
    for i, _ in enumerate(docs):
        with open(ti_user_path.format(i), 'r') as fp:
            user_list.append(json.load(fp))

    ti_list = []
    for i, _ in enumerate(docs):
        with open(ti_correct_path.format(i), 'r') as fp:
            ti_list.append(json.load(fp))

    dist_list = []
    for i, _ in enumerate(docs):
        with open(ti_topic_dist_dict.format(i), 'r') as fp:
            dist_list.append(json.load(fp))
try:
    output_dict_list = ef.run_merge_topics_ti(merge_ids_list, docs, ldamodel)
except:
    print("No topics are merged..")

# Calculation
try:
    log_odds = ef.calc_ti_log_odds(ti_choice, dist_list, docs, ti_list, user_list, ldamodel, output_dict_list=output_dict_list)
except NameError:
    log_odds = ef.calc_ti_log_odds(ti_choice, dist_list, docs, ti_list, user_list, ldamodel)
precision = ef.calc_ti_precision(ti_choice, ti_list, user_list)

end = time.time()
if __name__ == "__main__":
    print("Topic intrusion log odds: {}".format(log_odds))
    print("Topic intrusion precision: {}".format(precision))
    print("It took {0:.3g} mins".format((end-start)/60))
