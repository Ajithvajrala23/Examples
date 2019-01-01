
# coding: utf-8

# In[2]:

#Import al necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# In[3]:

#Get the data set
data = pd.read_csv("E:\Aegis\Kaggle\creditcardfraud/creditcard.csv")
df = data


# In[4]:

#Print the size of data set
print df.shape


# In[5]:

df.head()


# In[6]:

df.tail()


# In[7]:

#Check for missing values in the data
df.apply(lambda x: sum(x.isnull()))


# ---------------------------------------------------------
#           There are no missing values in the data
# ---------------------------------------------------------

# In[8]:

#check for categorical varaibles in the data
df.apply(lambda x: len(x.unique()))


# In[9]:

df.describe()


# _________________________________________________________________________________________________________________
# 
# 
# From the above analysis it is clear that 
#     1. There are no missing values in the data.
#     2. "Time" variable keeps on increasing from first observation to last observation.
#     3. "Amount" attribute contains higher range values when it is compared to other set of independent variables.
#         So we need to normalize the all the independent variables to same range if we are going to apply logistic model.
#     
#     
#     
# ___________________________________________________________________________________________________________________

# In[10]:

#First we drop the "Time" from our dataframe
df = df.drop(['Time'], axis=1)
df.head()


# In[11]:

#Now find correlation among the numerical variables
df_corr = df.copy()
del df_corr['Class']


# In[12]:

#Compute correlation matrix
corr = df_corr.corr()
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
#Generate a mask for upper  traingle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] =True

#Set up matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

#Generate a custom diverging color map
cmap= sns.diverging_palette(220, 10, as_cmap= True)


#Draw the heat map with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# In[13]:

corr


# # There is no correlation blw the input variables

# In[14]:

#We will analyse the output categorical variable
get_ipython().magic(u'matplotlib inline')
count_classes = pd.value_counts(df.Class, sort = True).sort_index()
print count_classes
print 
#Plot the output
count_classes.plot(kind="bar")
plt.title("Fraud transactions")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[15]:

from __future__ import division
base_line_accuracy = 284315/(284315+492)
print base_line_accuracy


# _______________________________________________________________________________________________________________
# Following points are clear from the above analysis.
#     1. base_line_accuracy itself measures to be 99.8%
#     2. It is very much clear that dataset is highly imbalanced.
#     3. So dataset needs be "undersampled" or "oversampled" to make sure that output labels are of same number.
# 
# _________________________________________________________________________________________________________________

# In[16]:

print count_classes


# Approach1:
#     
# #______Under sampling Technique____#
# 
# In this approach we randomly take 492 samples from "Class variable 0" (Since Class 1 has less no.of variables)
# 

# In[17]:

no_frauds = len(df[df.Class ==1])
print "No.of fradulent transactions are : ", no_frauds

#Find the index related to fraud transactions
fraud_index = np.array(df[df.Class ==1].index)

#now take the indices of normal transactions
normal_indices = df[df.Class ==0].index

#Now pick the random samples 

random_normal_indices = np.random.choice(normal_indices, no_frauds, replace=False)
random_normal_indices = np.array(random_normal_indices)

#Now we combine both random indices and fradulent indices

appended_indices = np.concatenate([fraud_index, random_normal_indices])

#Now using indices get the data from those indices

appended_data_set = df.iloc[appended_indices, :]
appended_data_set.head()


# In[18]:

appended_data_set.shape


# In[19]:

under_sample_count_classes = pd.value_counts(appended_data_set.Class, sort = True).sort_index()
print under_sample_count_classes


# In[20]:

under_sample_count_classes.plot(kind="bar")
plt.title("Fraud transactions plot after Under sampling")
plt.xlabel("Class")
plt.ylabel("Frequency")


# #Now we have equal no.of class labels

# In[21]:

#We now differentiate the input attributes and output labels
X_under = appended_data_set.ix[:, df.columns != "Class"]
Y_under = appended_data_set.ix[:, df.columns == "Class"]


#Now we normalize all the input variables to same range
import math
X_norm_under = (X_under - X_under.mean())/(X_under.max() - X_under.min())
#
X_norm_under.head()


# All the preprocessing steps are completed.
# 
# Now we have split the data into training_set and test_set.

# In[22]:

from sklearn.cross_validation import train_test_split
X_norm_under_train,X_norm_under_test,Y_norm_under_train,Y_norm_under_test = train_test_split(X_norm_under, Y_under, test_size = 0.3, random_state =0)
print "Size of training dataset: ", len(X_norm_under_train)
print "Size of test dataset: ", len(X_norm_under_test)


# In[23]:

print "The no.of class labels in training set: "
print pd.value_counts(Y_norm_under_train.Class, sort = True).sort_index()
print "The no.of class labels in test set : "
print pd.value_counts(Y_norm_under_test.Class, sort = True).sort_index()


# In[24]:

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.linear_model import LogisticRegression

def printing_Kfold_scores(x_train, y_train):
    fold = KFold(len(y_train), 5, shuffle=False)
    #
    #Defining C-parameters
    
    c_parm_range = [0.01,0.1,1,10,100]
    
    results_table= pd.DataFrame(index=range(len(c_parm_range),2), columns=['C_Parameter','Mean recall score'])
    
    results_table['C_Parameter'] = c_parm_range
    
    j=0
    for c_parm in c_parm_range:
        print "___________________________________________"
        print "C parameter: ", c_parm
        print "___________________________________________"
        print ""
        recall_accs =[]
        for iteration, indices in enumerate(fold, start=1):
            #Call logistic with each c-parameter
            lr = LogisticRegression(C= c_parm, penalty='l1')
            
            lr.fit(x_train.iloc[indices[0],:], y_train.iloc[indices[0],:].values.ravel())
            
            y_pred_undersample=lr.predict(x_train.iloc[indices[1],:].values)
            
            recall_acc = recall_score(y_train.iloc[indices[1],:].values, y_pred_undersample)
            
            recall_accs.append(recall_acc)
            
            print "Iteration : ", iteration, " : recall score = ",recall_acc
        
        results_table.ix[j, "Mean recall score"] = np.mean(recall_acc)
        
        j+=1
        
        print "",
        print "Mean recall score ", np.mean(recall_accs)
        print "",
    
    best_c = results_table.loc[results_table["Mean recall score"].idxmax()]['C_Parameter']
    print "_____________________________________________________________________________________"
    print "Best model with c-parm is ", best_c
    
    return best_c


# In[25]:

best_c = printing_Kfold_scores(X_norm_under_train, Y_norm_under_train)


# In[26]:

from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(C=100, penalty= 'l1')
lr1.fit(X_norm_under_train, Y_norm_under_train)
Y_predict_lr1 = lr1.predict(X_norm_under_test)


# In[27]:

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
cnf_matrix = confusion_matrix(Y_norm_under_test, Y_predict_lr1)
cnf_matrix


# In[28]:

#Plot confusion matrixri
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max()/2.
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("true label")
    plt.xlabel("predictedc label")


# In[29]:

np.set_printoptions(precision=2)
print "accuracy : ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names =[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title= "Confusion Matrix")
plt.show()


# In[30]:

#AUC
y_pred_undersample_score = lr1.fit(X_norm_under_train, Y_norm_under_train.values.ravel()).decision_function(X_norm_under_test.values)
fpr, tpr, thresholds = roc_curve(Y_norm_under_test, y_pred_undersample_score)
roc_auc = auc(fpr, tpr)

#plot ROC
plt.title("Receiver Operating characteristics")
plt.plot(fpr, tpr, 'b', label="AUC= %0.2f" %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


# In[32]:

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(random_state=0)
clf1.fit(X_norm_under_train,Y_norm_under_train)


# In[33]:

from __future__ import division
y_under_pred = clf1.predict(X_norm_under_test)
print confusion_matrix(Y_norm_under_test, y_under_pred)
cnf_matrix = confusion_matrix(Y_norm_under_test, y_under_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()


# In[45]:

# SVM 
from sklearn import svm


# In[46]:

clf = svm.SVC()
clf.fit(X_norm_under_train,Y_norm_under_train)


# In[47]:

from __future__ import division
y_under_pred = clf.predict(X_norm_under_test)
print confusion_matrix(Y_norm_under_test, y_under_pred)
cnf_matrix = confusion_matrix(Y_norm_under_test, y_under_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()


# In[48]:

clf1 = svm.SVC(kernel='linear')
clf1.fit(X_norm_under_train,Y_norm_under_train)


# In[49]:

from __future__ import division
y_under_pred = clf1.predict(X_norm_under_test)
print confusion_matrix(Y_norm_under_test, y_under_pred)
cnf_matrix = confusion_matrix(Y_norm_under_test, y_under_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# Approach 2
# #_______________Over Sampling_______________#

# In[34]:

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[35]:

#Over sampling using smote
X = df.ix[:, df.columns != "Class"]
Y = df.ix[:, df.columns == "Class"]
#
X_smote_train, X_smote_test, Y_smote_train, Y_smote_test= train_test_split(X,Y, test_size=0.3, random_state = 0)
oversampler = SMOTE(random_state=0)
x_os, y_os = oversampler.fit_sample(X_smote_train, Y_smote_train)


# In[36]:

print "No.od variables with class labels as 1: ", len(y_os[y_os==1])
print "No.of variables with class labels as 0: ", len(y_os[y_os==0])


# In[37]:

count_classes_smote = pd.value_counts(y_os, sort = True).sort_index()
count_classes_smote.plot(kind="bar")
plt.title("Fraud transactions after Over Sampling")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[38]:

#Random Forest classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(x_os,y_os)


# In[39]:

from __future__ import division
y_smote_pred = clf.predict(X_smote_test)
print confusion_matrix(Y_smote_test, y_smote_pred)
cnf_matrix = confusion_matrix(Y_smote_test, y_smote_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()


# In[40]:

from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(C=100, penalty= 'l1')
lr1.fit(x_os, y_os)
y_smote_pred = lr1.predict(X_smote_test)


# In[43]:

print confusion_matrix(Y_smote_test, y_smote_pred)
cnf_matrix = confusion_matrix(Y_smote_test, y_smote_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()


# In[44]:

#AUC
y_pred_undersample_score = lr1.fit(x_os, y_os.ravel()).decision_function(X_smote_test.values)
fpr, tpr, thresholds = roc_curve(Y_smote_test, y_pred_undersample_score)
roc_auc = auc(fpr, tpr)

#plot ROC
plt.title("Receiver Operating characteristics")
plt.plot(fpr, tpr, 'b', label="AUC= %0.2f" %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


# In[50]:

#___________#
#SVM classifier
clf = svm.SVC()
clf.fit(x_os,y_os)


# In[51]:

from __future__ import division
y_smote_pred = clf.predict(X_smote_test)
print confusion_matrix(Y_smote_test, y_smote_pred)
cnf_matrix = confusion_matrix(Y_smote_test, y_smote_pred)
np.set_printoptions(precision=2)
print "recall Metric in the test set: ", cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion Matrix")
plt.show()

