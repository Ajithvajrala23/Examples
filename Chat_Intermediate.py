
# coding: utf-8

# In[1]:

#Get the chat history
#
import pandas as pd
Aegis_Domain=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Aegis_Domain.csv")
Data_Mining_General=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Data_Mining_General.csv")
Pythom_R_ML=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Pythom_R_ML.csv")
Statistics=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Statistics.csv")
Hadoop=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Hadoop.csv")
Visualization=pd.read_csv("F:\Aegis school of business and telecommunication\Project\Data_set/Visualization.csv")
##
#Size of each data set
print Aegis_Domain.shape
print Data_Mining_General.shape
print Pythom_R_ML.shape
print Statistics.shape
print Hadoop.shape
print Visualization.shape

#
#print df
#
#Pre processing the data
import re
def cleaning_text(review):
    cleaned =re.sub("[^a-zA-Z/:.?]", " ",review)
    words =cleaned.lower().split()
    return(" ".join( words ))
new_words=[]
for i in range(0,Aegis_Domain.size):
    new_words.append(cleaning_text(Aegis_Domain["QA"][i]))
for i in range(0,Data_Mining_General.size):
    new_words.append(cleaning_text(Data_Mining_General["QA"][i]))
for i in range(0,Pythom_R_ML.size):
    new_words.append(cleaning_text(Pythom_R_ML["QA"][i]))
for i in range(0,Statistics.size):
    new_words.append(cleaning_text(Statistics["QA"][i]))
for i in range(0,Hadoop.size):
    new_words.append(cleaning_text(Hadoop["QA"][i]))
for i in range(0,Visualization.size):
    new_words.append(cleaning_text(Visualization["QA"][i]))

#print new_words


# In[2]:

#Import the chatbot 

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
#
#Import logic adapters for basic operations
#
chatbot=ChatBot("Example1",logic_adapters=[
        "chatterbot.adapters.logic.ClosestMatchAdapter",
        "chatterbot.adapters.logic.MathematicalEvaluation",
        "chatterbot.adapters.logic.TimeLogicAdapter"])
#
#Import List trainer for training the chat history
#
from chatterbot.trainers import ListTrainer
#
#Move the preprocessed chat history to conversations
#
conversation=new_words
chatbot.set_trainer(ListTrainer)
#
#training the conversation
#
chatbot.train(conversation)
#
#Import corpus.english for basic greetings and casual replies
#
chatbot.train("chatterbot.corpus.english")
chatbot.set_trainer(ChatterBotCorpusTrainer)
#chatbot.train("chatterbot.corpus.english")
chatbot.train(
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations"
)


# In[3]:

#Give the input and get the response from the chat bot
#
print chatbot.get_response("Tell me about business Analytics?")
print chatbot.get_response("what exactly data science is?")
print chatbot.get_response("where is campus")
print chatbot.get_response("can i get a study loan")
print chatbot.get_response("What is Aegis")
print chatbot.get_response("what is your name?")
print chatbot.get_response("good morning")
print chatbot.get_response("What is Hadoop?")
print chatbot.get_response("What are course timings?")
print chatbot.get_response("where is difference between python and R?")
print chatbot.get_response("what is data mining?")



# In[ ]:




# In[ ]:



