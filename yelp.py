
# coding: utf-8

# In[1]:

import pandas as pd
import pickle


# In[2]:

with open('E:\Aegis\NLP\Project/yelp_data_subset.pkl', 'rb') as f:
    data = pickle.load(f)


# In[3]:

dtest = data


# In[4]:

#Converting list to array
import numpy as np
myarray = np.asarray(dtest)


# In[5]:

#myarray


# In[6]:

#Converting list to data frame using Pandas

df = pd.DataFrame(dtest, columns=['votes', 'user_id', 'review_id', 'text','business_id','date','type'])


# In[7]:

df.head()


# In[8]:

df.shape


# In[9]:

#Data extraction is completed.
#
#Check for missing values in the data
#
df.apply(lambda x: sum(x.isnull()))


# In[10]:

#There are no missing values in the data


# In[11]:

df_ = df


# In[12]:

del df_['votes']
#Look at no.of categorical variables in the data
df_.apply(lambda x: len(x.unique()))


# In[13]:

#____Classification of reviews_____#
#Represent the text in numerical attributes


# In[14]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer(max_df = 0.5, min_df= 2, stop_words='english')


# In[15]:

Reviews_text = df_['text']


# In[16]:

X = vectorizer.fit_transform(Reviews_text)
X


# In[17]:

from sklearn.cluster import KMeans
km = KMeans(n_clusters= 5, init='k-means++', max_iter = 100, n_init=1, verbose=True)


# In[18]:

km.fit(X)


# In[19]:

import numpy as np
np.unique(km.labels_, return_counts=True)


# In[20]:

#Remove the same words present in among all the cluster and See the top 10 words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk


# In[21]:

_stopwords = set(stopwords.words('english') + list(punctuation) + ["'s","''"])


# In[22]:

text ={}
for i, cluster in enumerate(km.labels_):
    oneDocument = Reviews_text[i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument


# In[23]:

keywords = {}
counts ={}
for cluster in range(5):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq= FreqDist(word_sent)
    keywords[cluster] =nlargest(100, freq, key = freq.get)
    counts[cluster] = freq


# In[24]:

unique_keys ={}
for cluster in range(5):
    other_clusters = list(set(range(5)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique= set(keywords[cluster]) -keys_other_clusters
    unique_keys[cluster] = nlargest(100, unique, key=counts[cluster].get)


# In[25]:

keywords[0]


# In[26]:

df_.head()


# In[83]:

df_['Cluster0'] = 0
df_['Cluster1'] = 0
df_['Cluster2'] = 0
df_['Cluster3'] = 0
df_['Cluster4'] = 0
#
df_['Clust0_Sent'] = "NEUTRAL"
df_['Clust1_Sent'] = "NEUTRAL"
df_['Clust2_Sent'] = "NEUTRAL"
df_['Clust3_Sent'] = "NEUTRAL"
df_['Clust4_Sent'] = "NEUTRAL"


# In[84]:

df_.head(2)


# In[29]:

length_df = len(df_['text'])
length_df


# In[30]:

print len(sent_tokenize(df_.text[0]))
print sent_tokenize(df_.text[0])[1]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

df_.head()


# In[ ]:

#df_.head(3)


# In[ ]:

df_.to_csv("yelp_int2.csv", encoding='utf-8')


# In[ ]:

df_.shape


# In[58]:

#unique_keys


# In[ ]:




# In[ ]:




# #Amazon Food reviews

# In[31]:

Amazon = pd.read_csv(r"E:\Aegis\NLP\Amazon_Food_review\amazon-fine-foods/Reviews.csv")
Amazon.shape


# In[32]:

Amazon_ = Amazon


# In[33]:

Amazon_.apply(lambda x: sum(x.isnull()))


# In[34]:

del Amazon_['ProfileName']
del Amazon_['HelpfulnessNumerator']
del Amazon_['HelpfulnessDenominator']
del Amazon_['Time']
del Amazon_['Summary']


# In[35]:

Amazon_.head()


# In[36]:

Amazon_['Sentiment'] ="A"


# In[37]:

Amazon_['Sentiment'] [Amazon_['Score'] >= 4 ] ="positive"
Amazon_['Sentiment'] [Amazon_['Score'] == 3 ] ="neutral"
Amazon_['Sentiment'] [Amazon_['Score'] <= 2 ] ="negative"


# In[38]:

Amazon_.head()


# In[39]:

Amazon_['Sentiment'][Amazon_['Score'] == 5].count()


# In[40]:

print Amazon_['Sentiment'][Amazon_['Sentiment'] == "negative"].count()
print Amazon_['Sentiment'][Amazon_['Sentiment'] == "neutral"].count()


# In[41]:

fixed_length =10000


# In[42]:

import numpy as np


# In[43]:

positive_indices = np.array(Amazon_[Amazon_.Sentiment=="positive"].index)
negative_indices = np.array(Amazon_[Amazon_.Sentiment=="negative"].index)
#normal_indices = Amazon_[Amazon_['Sentiment']=="negative"].index
#now pick random samples
random_pos_indices = np.random.choice(positive_indices, fixed_length, replace=False)
random_neg_indices = np.random.choice(negative_indices, fixed_length, replace=False)
random_pos_indices = np.array(random_pos_indices)
random_neg_indices = np.array(random_neg_indices)

#Now Combine both indices
appended_indices = np.concatenate([random_pos_indices, random_neg_indices])

#
appended_dataset = Amazon_.iloc[appended_indices,:]
appended_dataset.head()


# In[44]:

pd.value_counts(appended_dataset.Sentiment, sort=True).sort_index()


# In[45]:

from bs4 import BeautifulSoup  
import nltk
import re
from nltk.corpus import stopwords


# In[46]:

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 


# In[47]:

print appended_dataset.shape


# In[48]:

print appended_dataset["Text"].size
appended_dataset.iloc[4]['Text']


# In[49]:

num_reviews = appended_dataset["Text"].size
clean_train_reviews = []

for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( appended_dataset.iloc[i]["Text"] ) )


# In[50]:

from sklearn.feature_extraction.text import CountVectorizer
# 
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 
#
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
#train_data_features = train_data_features.toarray()


# In[51]:

vocab = vectorizer.get_feature_names()
print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, appended_dataset["Sentiment"] )


# In[63]:

clean_test_reviews = [] 
clean_review = review_to_words(     sent_tokenize(df_['text'][1])[0]     )
clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
print result


# In[85]:

length_df


# In[ ]:




# In[86]:

#Cluster0

for i in range(length_df):
    new_sent =[]
    for j in  range(len(sent_tokenize(df_['text'][i]))):
        #
        new_sent = sent_tokenize(df_.text[i])[j]
        new_words = [x.lower() for x in word_tokenize(new_sent)]
        
        for k in range(len(unique_keys[0])):
            if unique_keys[0][k] in new_words:
                df_['Cluster0'][i] = 1
                break
                break
        clean_test_reviews = []     
        result = []
    
    if df_['Cluster0'][i] == 1:
            
        clean_test_reviews = [] 
        clean_review = review_to_words(new_sent)
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()
        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        #print result
        df_['Clust0_Sent'][i] = result


# In[88]:

#Cluster1

for i in range(length_df):
    new_sent =[]
    for j in  range(len(sent_tokenize(df_['text'][i]))):
        #
        new_sent = sent_tokenize(df_.text[i])[j]
        new_words = [x.lower() for x in word_tokenize(new_sent)]
        
        for k in range(len(unique_keys[1])):
            if unique_keys[1][k] in new_words:
                df_['Cluster1'][i] = 1
                break
                break
             
        
    clean_test_reviews = []
    result = []
    if df_['Cluster1'][i] == 1:
            
        clean_test_reviews = [] 
        clean_review = review_to_words(new_sent)
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()
        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        #print result
        df_['Clust1_Sent'][i] = result


# In[87]:

for i in range(length_df):
    new_sent =[]
    for j in  range(len(sent_tokenize(df_['text'][i]))):
        #
        new_sent = sent_tokenize(df_.text[i])[j]
        new_words = [x.lower() for x in word_tokenize(new_sent)]
        
        for k in range(len(unique_keys[2])):
            if unique_keys[2][k] in new_words:
                df_['Cluster2'][i] = 1
                break
                break
        clean_test_reviews = []     
        result = []
    
    if df_['Cluster2'][i] == 1:
            
        clean_test_reviews = [] 
        clean_review = review_to_words(new_sent)
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()
        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        #print result
        df_['Clust2_Sent'][i] = result


# In[89]:

for i in range(length_df):
    new_sent =[]
    for j in  range(len(sent_tokenize(df_['text'][i]))):
        #
        new_sent = sent_tokenize(df_.text[i])[j]
        new_words = [x.lower() for x in word_tokenize(new_sent)]
        
        for k in range(len(unique_keys[3])):
            if unique_keys[3][k] in new_words:
                df_['Cluster3'][i] = 1
                break
                break
    clean_test_reviews = []     
    result = []
    
    if df_['Cluster3'][i] == 1:
            
        clean_test_reviews = [] 
        clean_review = review_to_words(new_sent)
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()
        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        #print result
        df_['Clust3_Sent'][i] = result


# In[90]:

for i in range(length_df):
    new_sent =[]
    for j in  range(len(sent_tokenize(df_['text'][i]))):
        #
        new_sent = sent_tokenize(df_.text[i])[j]
        new_words = [x.lower() for x in word_tokenize(new_sent)]
        
        for k in range(len(unique_keys[4])):
            if unique_keys[4][k] in new_words:
                df_['Cluster4'][i] = 1
                break
                break
    clean_test_reviews = []     
    result = []
    
    if df_['Cluster4'][i] == 1:
            
        clean_test_reviews = [] 
        clean_review = review_to_words(new_sent)
        clean_test_reviews.append( clean_review )

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()
        # Use the random forest to make sentiment label predictions
        result = forest.predict(test_data_features)
        #print result
        df_['Clust4_Sent'][i] = result


# In[91]:

df_.to_csv("yelp_final.csv", encoding='utf-8')


# In[92]:

df_.head(2)


# In[94]:

df_.tail()


# In[98]:

pd.value_counts(df_.business_id, sort = True)


# In[102]:

print pd.value_counts(df_['Clust0_Sent'][df_['business_id'] == "2SwC8wqpZC4B9iFVTgYT9A"])
print pd.value_counts(df_['Clust1_Sent'][df_['business_id'] == "2SwC8wqpZC4B9iFVTgYT9A"])
print pd.value_counts(df_['Clust2_Sent'][df_['business_id'] == "2SwC8wqpZC4B9iFVTgYT9A"])
print pd.value_counts(df_['Clust3_Sent'][df_['business_id'] == "2SwC8wqpZC4B9iFVTgYT9A"])
print pd.value_counts(df_['Clust4_Sent'][df_['business_id'] == "2SwC8wqpZC4B9iFVTgYT9A"])


# In[106]:

pd.value_counts(df_.Clust_Sent)


# In[108]:

print pd.value_counts(df_['Clust0_Sent'][df_['business_id'] == "IxQ1ATP_Wg_QujO9nywzcQ"])
print pd.value_counts(df_['Clust1_Sent'][df_['business_id'] == "IxQ1ATP_Wg_QujO9nywzcQ"])
print pd.value_counts(df_['Clust2_Sent'][df_['business_id'] == "IxQ1ATP_Wg_QujO9nywzcQ"])
print pd.value_counts(df_['Clust3_Sent'][df_['business_id'] == "IxQ1ATP_Wg_QujO9nywzcQ"])
print pd.value_counts(df_['Clust4_Sent'][df_['business_id'] == "IxQ1ATP_Wg_QujO9nywzcQ"])
