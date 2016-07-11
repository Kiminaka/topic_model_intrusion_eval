import evaluation_function as ef
import pandas as pd
import numpy as np
import gensim
from collections import Counter
import copy
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

print("Calculating...")

try:
    print('Merging predetermined topics..')
    merge_ids_list = config.merge_ids_list
except:
    print('No merging topics..')


# Read dataframe in pandas and create a list of docs
start = time.time()
docs = list(pd.read_csv(docs_path, index_col=0, header=None, encoding = "ISO-8859-1").values[1:])

# Read gensim model and dictionary
ldamodel = gensim.models.LdaModel.load(model_path)
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
topics = ldamodel.show_topics(num_topics=num_topics, num_words=100, formatted=False)

# create word intrusion datasets
try:
    # print('merging topics')
    wi_topic_dist_dict = ef.run_merge_topics(topics, merge_ids_list)
    wi_intrusion_dict = ef.create_word_intruder(wi_topic_dist_dict)
    wi_merged_dict = ef.merge_sets_wi(wi_topic_dist_dict, wi_intrusion_dict)
except:
    # print('No merging topics')
    wi_topic_dist_dict = copy.deepcopy(ef.top_n_words_from_topics(topics, 5))
    wi_intrusion_dict = ef.create_word_intruder(wi_topic_dist_dict)
    wi_merged_dict = ef.merge_sets_wi(wi_topic_dist_dict, wi_intrusion_dict)

# put in pandas DF
wi_topic_dist_dict = pd.DataFrame(list(wi_topic_dist_dict.values()))
wi_intrusion_dict = pd.DataFrame(list(wi_intrusion_dict.values()))
wi_merged_dict = pd.DataFrame(list(wi_merged_dict.values()))
print('wi data created!')

# save wi data
wi_intrusion_dict.to_csv(wi_correct_filename)
wi_merged_dict.to_csv(wi_merged_set_filename)

################################################################################################
# create topic intrusion datasets
try: # create data with predetermined topics to be merged
    output_dict_list = ef.run_merge_topics_ti(merge_ids_list, docs, ldamodel)
    topic_list, ti_list, merged_ti, topic_dist_list  = ef.create_ti_data_from_docs(docs, topics, ldamodel, output_dict_list, merge_ids_list)
except:
    print('failed')
    topic_list, ti_list, merged_ti, topic_dist_list  = ef.create_ti_data_from_docs(docs, topics, ldamodel)

print('ti data created!')

# save ti_list and merged_ti
if ti_choice == 'reusable':
    # create reusable training data for ti
    merge_ti_reusable = []
    ti_list_reusable = []
    for x,y in zip(merged_ti, ti_list):
        merge_ti_reusable.append(list(x.keys()))
        ti_list_reusable.append(list(y.keys())[0])

    merge_ti_reusable = pd.DataFrame(merge_ti_reusable)
    ti_list_reusable = pd.DataFrame(ti_list_reusable)

    for x in merge_ti_reusable:
        merge_ti_reusable[x] = merge_ti_reusable[x].apply(ef.topic_converter)

    for x in ti_list_reusable:
        ti_list_reusable[x] = ti_list_reusable[x].apply(ef.topic_converter)

    merge_ti_reusable.to_csv(ti_merged_set_path_reusable)
    ti_list_reusable.to_csv(ti_correct_path_reusable)

    for i, x in enumerate(topic_dist_list):
        with open(ti_topic_dist_dict.format(i), 'w') as fp:
            json.dump(dict(x), fp)

if ti_choice ==  'not_reusable_but_debuggable':
    for i, x in enumerate(ti_list):
        with open(ti_correct_path.format(i), 'w') as fp:
            json.dump(x, fp)

    for i, x in enumerate(merged_ti):
        with open(ti_merged_set_path.format(i), 'w') as fp:
            json.dump(x, fp)

    for i, x in enumerate(topic_dist_list):
        with open(ti_topic_dist_dict.format(i), 'w') as fp:
            json.dump(dict(x), fp)
