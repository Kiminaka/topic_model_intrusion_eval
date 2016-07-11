# docs_path = 'data/docs.csv' # original documents
docs_path = 'data/nyt.csv' # original documents
# model_path = 'data/model/stem_topic12.gensim' # make sure to have .gensim.state file in the same folder as well
# dictionary_path = 'data/model/stem_topic12.dict' #gensim dictionary

model_path = 'data/model/nyt_articles_topic10.gensim' # make sure to have .gensim.state file in the same folder as well
dictionary_path = 'data/model/nyt_articles_topic10.dict' #gensim dictionary


wi_user_filename = 'data/wi/word_intrusion_user_set.csv' #you will create the data. Read the doc for formating.
wi_correct_filename = 'data/wi/word_intrusion_correct_set.csv' #word intrusion sets. Will be generated automatically
wi_merged_set_filename = 'data/wi/wi_merged_df.csv' # merged sets for evaluators to test. Will be generated automatically

ti_user_path = 'data/ti/user/{}.json' #you will create the data. Read the doc for formating.
ti_user_path_reusable = 'data/ti/user/user_reusable.csv' #you will create the data. Read the doc for formating.
ti_correct_path = 'data/ti/intruder/{}.json' #word intrusion sets. Will be generated automatically
ti_correct_path_reusable = 'data/ti/intruder/intruder_reusable.csv' #word intrusion sets. Will be generated automatically
ti_merged_set_path = 'data/ti/merged/{}.json' # merged sets for evaluators to test. Will be generated automatically
ti_merged_set_path_reusable = 'data/ti/merged/merged_reusable.csv' # merged sets for evaluators to test. Will be generated automatically
ti_topic_dist_dict = 'data/ti/topic_dist/{}.json' # probabilities for each topics for docs. Will be generated automatically

num_topics = 10 # number of topics for your gensim model

""" If topic ids share same topic, the below is able to merge the topics.
If don't need to merge topics, comment out the merge_ids_list variable.
Example:
if topic 0,3,10,7 share a topic, create a list in a list. (The smallest id become the merged id. In this case 0 is the representative)
if additionaly topic 4,9 share a topic, add a list in a list
"""
# merge_ids_list = [[0,3,10,7],[4,9]]
merge_ids_list = [[0,9],[1,8],[2,6],[3,4],[5,7]]

""" setting "reusable" creates training sets with topics .
setting "not_reusable_but_debuggable" creates training sets with topic words.
make sure you have one from the two options.
"""
ti_choice = 'reusable'
# ti_choice = 'not_reusable_but_debuggable'

"""for "reusable" option, set up the topic names to display for evaluators. They will see the names
for 'not_reusable_but_debuggable' option, comment out "topic_names" variable
"""
# topic_names = {
#                 0:'no_topic&random_topic',
#                 1:'program&goal_discussion',
#                 2:'program&question&support',
#                 4:'meal&ingredient',
#                 5:'program&encouragement&update',
#                 6:'weight_goal_discussion',
#                 8:'meal_suggestion_discussion',
#                 11:'exercise'
#                 }

topic_names = {
                0:'world politics',
                1:'national politics',
                2:'art and leisure',
                3:'world business',
                5:'sports'
                }
