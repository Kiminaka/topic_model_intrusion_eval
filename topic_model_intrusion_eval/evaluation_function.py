from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from functools import reduce
import pandas as pd
import numpy as np
import gensim
import re
import copy
import operator
import random
import config

model_path = config.model_path # make sure to have .gensim.state file too
dictionary_path = config.dictionary_path

ldamodel = gensim.models.LdaModel.load(model_path)
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
num_topics = config.num_topics
try:
    topic_names = config.topic_names
except:
    print('Using non-reusable option..')

# preprocessing ###############################################################################################
def preprocessing(doc): #stop word as optional
        x = re.sub("[^a-zA-Z]", " ", doc) #only words
        x = x.lower().split()
        stemmer = SnowballStemmer("english") # use snowball
        stops = set(stopwords.words("english")) # set is faster than list
        x = [stemmer.stem(word) for word in x if word not in stops]
        return(x)

# create probability distribution
def topic_predictor(doc, ldamodel):
    tokens = preprocessing(doc)
    bow = dictionary.doc2bow(tokens)
    return sorted(ldamodel.get_document_topics(bow), key=lambda x:x[1], reverse=True) # order by probability

# word intrusion ###############################################################################################

# create list of words, limit num_words and shuffle the items
def top_n_words_from_topics(gensim_show_topics, num_words=5):
    topic_words_dict = dict()
    for i, topic in enumerate(gensim_show_topics):
        temp =[]
        for dist in topic[1]:
            temp.append(dist[0])
        temp = temp[:num_words]
        random.shuffle(temp)
        topic_words_dict[i] = temp
    return topic_words_dict

def merge_topics(topic_dist_dict, *topic_ids):
    """
    topic_words_dict = output of top_5_words_from_topic
    topic_ids = topics to merge. Example: [2,4,5]
    Note: the smallest id of merging topics become the merged topic id
    """
    output_dict = copy.deepcopy(topic_dist_dict)
    merged_list = []
    topic_ids = sorted(topic_ids)
    for topic_id in topic_ids:
        merged_list.append(output_dict[topic_id])
    output_dict[topic_ids[0]] = random.sample(set(reduce(operator.add,merged_list)), 5)
    for topic_id in topic_ids[1:]:
        del output_dict[topic_id]
    return output_dict

# run the merge_topics function iteratively to automatically merge multiple topics
def run_merge_topics(gensim_show_topics, topic_ids_list):
    topic_dist_dict = top_n_words_from_topics(gensim_show_topics, 5)
    for i, topic_ids in enumerate(topic_ids_list):
        if i == 0:
            output_topic_dist = merge_topics(topic_dist_dict, *topic_ids)
        else:
            output_topic_dist = merge_topics(output_topic_dist, *topic_ids)
    return output_topic_dist

# pick a word at random from topic words
def create_word_intruder(topics_dict):
    output_dict = dict()
    for topic_id in topics_dict.keys():
        output_dict[topic_id] = random.choice(topics_dict[topic_id])
    return output_dict

# merge topic sets with intruder set but with reversed order. Then shuffle the words
def merge_sets_wi(topics_dict, intrusion_dict):
    out_put_dict = copy.deepcopy(topics_dict)
    for i,ii in zip(out_put_dict.keys(),sorted(intrusion_dict.keys(), reverse=True)):
        out_put_dict[i].append(intrusion_dict[ii])
        random.shuffle(out_put_dict[i])
    return out_put_dict

# create evaluation set (list of intrusion words)
def validation_set_wi(word_intrusion_dict):
    return list(word_intrusion_dict.values())

# calculate word intrusion precision
def calc_wi_precision(word_intrusion_list, user_answer_list):
    return Counter([x==y for (x,y) in zip([x[0] for x in word_intrusion_list],[x[0] for x in user_answer_list])])[True]/len(word_intrusion_list)

# topic intrusion ###############################################################################################

# pick top x topics for doc
def top_topic_picker(topic_dist, max_x=3, num_topics=num_topics):
    topic_list = []
    for x in topic_dist[:max_x]:
        if x[1] > 1/num_topics:
            topic_list.append(x[0])
    return topic_list

# select x words for certain topics
def get_words_for_topics(topic_ids, show_topics, num_words=10):
    topic_words_dict = top_n_words_from_topics(show_topics, num_words)
    output = dict()
    for topic_id in topic_ids:
        output[topic_id] = topic_words_dict[topic_id]
    return output

def create_topic_intruder(topic_dist, show_topics, num_words=10, merge_ids_list=False):
    if merge_ids_list == False:
        intruder = topic_dist[-1][0]
    else:
        intruder = topic_dist[-1][0]
        for x in merge_ids_list:
            if intruder in x:

                intruder = sorted(x)[0]
    output = dict()
    topic_words_dict = top_n_words_from_topics(show_topics, num_words)
    output[intruder] = topic_words_dict[intruder]
    return output

# merge true set with intrusion topic
def merge_sets_ti(topics_dict, intrusion_dict):
    out_put_dict = copy.deepcopy(topics_dict)
    out_put_dict.update(intrusion_dict)
    return out_put_dict

# create evaluation set (list of intrusion topic words)
def validation_set_ti(topic_intrusion_dict):
    return list(topic_intrusion_dict.values())[0]

# create topic words dict, topic intrusion words dic and merged set
def create_ti_data_from_docs(list_of_docs, show_topics, ldamodel, output_dict_list=False, merge_ids_list=False, debugger=False):
    topic_dist_list = []
    topic_list = []
    ti_list = []
    merged_list = []
    for i, doc in enumerate(list_of_docs):
        if output_dict_list == False:
            topic_dist = topic_predictor(doc[0], ldamodel)
        else:
            topic_dist = list(output_dict_list[i].items())
            topic_dist = sorted(dict(topic_dist).items(), key=lambda x:x[1], reverse=True)
        if debugger == True:
            print(i,topic_dist)
        topic_dist_list.append(topic_dist)
        top_topics = top_topic_picker(topic_dist, max_x=3, num_topics=num_topics)
        topic_words = get_words_for_topics(top_topics, show_topics)
        topic_list.append(topic_words)
        if merge_ids_list == False:
            ti = create_topic_intruder(topic_dist, show_topics)
        else:
            ti = create_topic_intruder(topic_dist, show_topics, merge_ids_list=merge_ids_list)
        ti_list.append(ti)
        merged_list.append(merge_sets_ti(topic_words, ti))

    return topic_list, ti_list, merged_list, topic_dist_list

# merge predetermined topics
def merge_topics_ti(merge_ids ,docs, ldamodel, prob_dict = False):
    merge_ids = sorted(merge_ids)
    output_list = []

    for i, doc in enumerate(docs):
        if prob_dict == False:
            output_dict = dict(topic_predictor(list(doc)[0], ldamodel))
        else:
            output_dict = prob_dict[i]
        prob = 0
        for topic_id in list(output_dict.keys()):

            if topic_id in merge_ids:
                prob += output_dict[topic_id]
                output_dict[merge_ids[0]] = prob

        for delete_id in merge_ids[1:]:
            try:
                del output_dict[delete_id]
            except:
                continue
        output_list.append(output_dict)
    return output_list

# merge multiple topics iteratively
def run_merge_topics_ti(merge_topics_list, docs, ldamodel):
    for i, x in enumerate(merge_topics_list):
        if i == 0:
            output = merge_topics_ti(x ,docs, ldamodel)
        else:
            output = merge_topics_ti(x ,docs, ldamodel, output)
    return output

# log odds for topic intrusion
# compare top topic probability with intruder topic probability
def calc_ti_log_odds(ti_choise, topic_dists, docs_list, topic_intrusion_dict, user_answer_dict,
                    ldamodel, num_words=10, output_dict_list=None, merge_ids_list=None):
    numerator = 0
    for x, doc in enumerate(docs_list):
        if output_dict_list == None:
            topic_dist = dict(topic_predictor(doc[0], ldamodel)) # USE topic_dist_list created in create_datasets
            topic_dist = topic_dists[x]
            if ti_choise == 'not_reusable_but_debuggable':
                numerator += np.log(topic_dist[str(list(topic_intrusion_dict[x].keys())[0])]/
                      topic_dist[str(list(user_answer_dict[x].keys())[0])])

            if ti_choise == 'reusable':
                numerator += np.log(topic_dist[str(topic_intrusion_dict[x])]/
                      topic_dist[str(user_answer_dict[x])])

        else:
            topic_dist = output_dict_list[x]
            topic_dist = topic_dists[x]
            if ti_choise == 'not_reusable_but_debuggable':
                numerator += np.log(topic_dist[str(list(topic_intrusion_dict[x].keys())[0])]/
                      topic_dist[str(list(user_answer_dict[x].keys())[0])])

            if ti_choise == 'reusable':
                numerator += np.log(topic_dist[str(topic_intrusion_dict[x])]/
                      topic_dist[str(user_answer_dict[x])])
    return numerator/len(topic_intrusion_dict)


# precision for topic intrusion
def calc_ti_precision(ti_choise, topic_intrusion_dict, user_answer_dict):
    numerator = 0
    if ti_choise == 'not_reusable_but_debuggable':
        for topic, ti in zip(topic_intrusion_dict, user_answer_dict):
            if sorted(list(topic.values())[0]) == sorted(list(ti.values())[0]):
                numerator +=1
        return numerator/len(topic_intrusion_dict)

    if ti_choise == 'reusable':
        for topic, ti in zip(topic_intrusion_dict, user_answer_dict):
            if topic == ti:
                numerator +=1
        return numerator/len(topic_intrusion_dict)


# convert topic id numbers with topic names and reverse is possible with reverse == True
def topic_converter(topic_id, reverse=False):
    if reverse == False:
        try:
            return topic_names[topic_id]
        except KeyError:
            return topic_id
    if reverse == True:
        try:
            for original_id, topic_name in topic_names.items():
                if topic_name == topic_id:
                    return original_id
        except KeyError:
            return topic_id
