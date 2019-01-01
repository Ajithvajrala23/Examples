
# coding: utf-8

# In[1]:

#Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:

#_______ Data Exploration _________#

#import the training and test data set

train_df = pd.read_csv("E:\Aegis\Python\project/Train_UWu5bXk.csv")

test_df = pd.read_csv("E:\Aegis\Python\project/Test_u94Q5KV.csv")


# In[3]:

print "Training set size is: " , train_df.shape
print "Test Set size is: " , test_df.shape


# In[4]:

#Now make a new column of "train" in train data set and "test" in test data set
train_df['Type'] = "Train"
test_df['Type'] = "Test"


# In[5]:

#now join the training and test Data
data = pd.concat([train_df, test_df], ignore_index = True)
print train_df.shape, test_df.shape, data.shape


# In[6]:

#Now see the data 
print data.head()


# In[7]:

print data.tail()


# In[8]:

#_________ Data Cleaning _______#

# Missing Value Treatment:

#Now see the no.of missing values in the data set

data.apply(lambda x: sum(x.isnull()))


# In[9]:

#ignore Outlet sales because -for test set we don't have outletsales
#There are missing values for Item_weight and Outlet_Size


# In[10]:

#now perform summary steps on the data
print data.describe()


# In[11]:

#Look at no.of categorical variables in the data
data.apply(lambda x: len(x.unique()))


# In[12]:

# This shows the following are categorical variables
#   Item_Fat_Content          -> 5
#   Item_Type                 -> 16
#   Outlet_Establishment_Year -> 9
#   Outlet_Identifier         -> 10
#   Outlet_Location_Type      -> 3
#   Outlet_Size               -> 4
#   Outlet_Type               -> 4


# In[13]:

data.dtypes


# In[14]:

#filter the categorical variable
categorical = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
#Remove the index and Type of data set
categorical = [x for x in categorical  if x not in ['Item_Identifier','Outlet_Identifier','Type']]
#Now print the frequency of categorical variables
for col in categorical:
    print '\n Frequencies of categories of variable : %s'%col
    print data[col].value_counts()


# In[15]:

#Missing values imputation
#Determine the avg weight per item
item_avg_weight = data.pivot_table(values = 'Item_Weight', index = 'Item_Identifier')

#Check the index of missing values by getting their boolean variables
miss_bool = data['Item_Weight'].isnull()

#impute weights and check missing values before and after imputation
print 'Original Missing values: %d'%sum(miss_bool)

data.loc[miss_bool, 'Item_Weight'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])


print 'Final Missing values are : %d'%sum(data['Item_Weight'].isnull())


# In[16]:

#Check the data set now
data.describe()
#Everything looks good


# In[17]:

#Outlet_Size have missing values. So impute them with mode of outlet_size of particular type
from scipy.stats import mode

#Determine the mode for each one
outlet_size_mode = data.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x: mode(x).mode[0]))

print "Mode for each Outlet_type: ", outlet_size_mode

#Identify the index of missing values
miss_bool = data['Outlet_Size'].isnull()

#impute missing values with mode
print '\n Original Missing values: %d'%sum(miss_bool)

data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

print '\n After imputation missinng values: %d'%sum(data['Outlet_Size'].isnull())


# In[18]:

#Now final check whether there are any missing values
data.apply(lambda x: sum(x.isnull()))


# In[19]:

#item_identifier is given minimum value as 0, so change it
#Determine the avg visibility of the product
visibility_avg = data.pivot_table(values = 'Item_Visibility', index = 'Item_Identifier')

#impute 0 with mean visibility of that product

miss_bool = (data['Item_Visibility'] == 0)

print 'Number of 0 values present initially are: %d'%sum(miss_bool)

data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: visibility_avg[x])

print 'Number of 0 values after impuation: %d'%sum(data['Item_Visibility'] == 0)


# In[20]:

#As Visibility is more then item is more likey to be sold.
#So find the Avg_visibility of all stores and compare with Avg_visibility of each store

data['Item_Visibility_MeanRatio'] = data.apply(lambda x:x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis = 1)
print data['Item_Visibility_MeanRatio'].describe()


# In[21]:

#Since there are 16 categories of Item_Identifier, They are classified among Food, Drinks, Non-Consumable in different ways
#So bring down them to 3 categories
#Get the first two characters of ID
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#
#Now rename them as follows
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food',
                                                             'NC': 'Non-Consumable',
                                                             'DR': 'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[23]:

#_________ Feature Engineering ______#

#Since year is mentioned in the data set
#Caluclate the no of years from now

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

data['Outlet_Years'].describe()


# In[24]:

#There are mis-typed spellings in Item_Fat_Content
#So make everything under similar types
print "Originally present names: "
print data['Item_Fat_Content'].value_counts()

print "\nModified names: "
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat',
                                                             'reg': 'Regular',
                                                             'low fat': 'Low Fat'})

print data['Item_Fat_Content'].value_counts()


# In[25]:

#There is Non-Consumable Item, so remove fat-content data for that item
data.loc[data['Item_Type_Combined'] =="Non-Consumable", 'Item_Fat_Content'] ='Non-Edible' 
data['Item_Fat_Content'].value_counts()


# In[26]:

# Scikit-learn accepts only numerical variables
# So convert the categorical variables in the data to numerical variables
#import library

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#create a new variable outlet

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']

le = LabelEncoder()

for i in var:
    data[i] = le.fit_transform(data[i])


# In[27]:

#This will also generate some dummy variables. So remove them

data = pd.get_dummies(data, columns =['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                      'Item_Type_Combined', 'Outlet'])


# In[28]:

data.dtypes


# In[29]:

# Take a look at Item_Type_Combined

data[['Item_Type_Combined_0', 'Item_Type_Combined_1', 'Item_Type_Combined_2']].head(5)


# In[30]:

#___________ Exporting Data___________#

#Now drop the columns which have  been converted

data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

#Divide back Train and Test data sets

train = data.loc[data['Type'] == "Train"]
test = data.loc[data['Type'] == "Test"]

#Drop the columns which are not required

test.drop(['Item_Outlet_Sales', 'Type'], axis =1, inplace = True)
train.drop(['Type'], axis = 1, inplace = True)

#Export files as train.csv and test.csv

train.to_csv("train_mod.csv", index = False)
test.to_csv("test_mod.csv", index = False)


# In[31]:

# Training data
train.head()


# In[32]:

#_______ Model Building _________#

#Create a function for all models

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

from sklearn import cross_validation, metrics

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    
    #fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
    
    #predict the training data set
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    #perform cross validation
    
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv =20, scoring = 'mean_squared_error')
    
    cv_score = np.sqrt(np.abs(cv_score))
    
    #print the model report
    
    print "\n Report "
    
    print "RMSE: %.4g" %np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    
    print "CV Score:  Mean- %.4g | std- %.4g | Min- %.4g | Max- %.4g" %(np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    
    #Predict the test dataset
    
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export the submission file
    
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index= False)


# In[44]:

#Now run the Linear-Regression model

from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictors = [x for x in train.columns if x not in [target] + IDcol]

#print predictions

alg1 = LinearRegression(normalize = True)

modelfit(alg1, train, test, predictors, target, IDcol, 'Linear_Output1.csv')

#print alg1.coef_

get_ipython().magic(u'matplotlib inline')

coef1 = pd.Series(alg1.coef_, predictors).sort_values()

coef1.plot(kind = 'bar', title ='Model Coefficients ')


# In[45]:

# Since coefficients are very large use Ridge Regression Model

predictors = [x for x in train.columns if x not in [target] + IDcol]

alg2 = Ridge(alpha = 0.05, normalize = True)

modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')

coef2 = pd.Series(alg2.coef_, predictors).sort_values()

coef2.plot(kind = 'bar', title = "Model Coefficients")


# In[47]:

#Decission Tree Model

from sklearn.tree import DecisionTreeRegressor

predictors = [ x for x in train.columns if x not in [target]+IDcol]

alg3 = DecisionTreeRegressor(max_depth = 15, min_samples_leaf = 100)

modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')

coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending = False)

coef3.plot(kind = 'bar', title = 'Feature Importances')


# In[48]:

#Since here there 4 variables with high importance, use these

predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth = 8, min_samples_leaf = 150)

modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')

coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending = False)

coef4.plot(kind ='bar', title = 'Feature Importance')


# In[51]:

# Random Forest

from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target]+IDcol]

alg5 = RandomForestRegressor(n_estimators = 200, max_depth =5, min_samples_leaf =100, n_jobs = 4)

modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')

coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending = False)

coef5.plot(kind = 'bar', title = "Feature Importances")


# In[52]:

# Try another Random Forest Model

#from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target]+IDcol]

alg6 = RandomForestRegressor(n_estimators = 400, max_depth =6, min_samples_leaf =100, n_jobs = 4)

modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')

coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending = False)

coef5.plot(kind = 'bar', title = "Feature Importances")
