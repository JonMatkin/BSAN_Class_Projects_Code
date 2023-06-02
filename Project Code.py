    # -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:28:32 2022

@author: matkij

This is my code for the term long project. I am working on the Spaceship Titanic
Compeition on Kaggle. I have to write a classifier to determine which passengers
were transported to another dimension based on a variety of numeric and 
categorical features.

Here is a link to the competition:
    
    https://www.kaggle.com/competitions/spaceship-titanic
    

"""

'''
12/2/2022 10:48
6 days before submission deadline

Here is what needs to get done:
    
    I have to implement different train-test split percentages and come up
    with different accuracies and ROC curve / precision-recall graphs.
    
    A big thing here will actually be calculating the ROC stuff.
    
    Should not be too bad but just needs to get done
    
    Produce graphs for all to be used in the write up perhaps
'''


# Import the necessary modules and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import the training and testing data

train = pd.read_csv('C:/Users/matkij/Documents/Machine Learning/train.csv')
test = pd.read_csv('C:/Users/matkij/Documents/Machine Learning/test.csv')
# Initial look at the data set
#print(train.head())
# Description of the data set with shape
# print(train.describe())
# Look at the types for each column of the data
#print(train.dtypes)
# Info on the data
#print(train.info())
# Looking at null values in the data
#print(train.isna().sum())

# Now to fill the NaN values. They will be filled with zero
train = train.fillna(0)

# Put class label into its own variable
y_train = train['Transported']
X_train = train.drop(columns=['Transported'])
'''
# Going to drop the name columns due to little value
X_train=X_train.drop(columns=['Name'])


The cabin variable is broken into three distinct parts:
    Deck Number
    Room Number
    Side of ship
So break down the cabin variable into these three parts
'''
X_train[['Deck Number', 'Room Number', 'Ship Side']] = train['Cabin'].str.split('/', expand=True)
# Now to drop the entire cabin feature
X_train=X_train.drop(columns=['Cabin'])
'''
I was having some issues with the one hot encoding. It ended up doing the encoding
for every room number which I am not sure is helpful. I think I will pull out some
of the categorical variables, do the one hot encoding for them, then put them back
into the main dataframe.
'''
X_train = X_train.drop(columns=['Name'])
train_categorical = X_train.drop(columns = ['PassengerId','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Room Number'])
# Also need to drop those columns from X_train
X_train = X_train.drop(columns=['HomePlanet','CryoSleep','Destination','VIP','Deck Number','Ship Side'])

# Change the room number from an object to a numeric
X_train['Room Number'] = pd.to_numeric(X_train['Room Number'])

# Change passenger id and group size into their own columns and then drop Id
X_train[['Id','Group Size']] = X_train['PassengerId'].str.split('_',expand=True)
X_train = X_train.drop(columns=['Id'])
X_train['Group Size'] = pd.to_numeric(X_train['Group Size'])

'''
I need to scale the following variables:
    Age
    RoomService
    FoodCourt
    ShoppingMall
    Spa
    VRDeck
    Group Size
    
The following code will do that. I will use StandardScaler for it
'''

X_train[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck', 'Group Size']] = StandardScaler().fit_transform(X_train[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Group Size']])

# Preform one-hot encoding on the categorical variables
train_cat_encoded = pd.get_dummies(train_categorical)
# Turn the encodings into numerical values
train_cat_encoded = train_cat_encoded.astype(float)
# Reset indices for both dataframes
train_cat_encoded.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)
# Put both dataframes back together
X_train_enc = pd.concat([X_train,train_cat_encoded], axis=1)

# Get rid of the PassengerId column
X_train_enc = X_train_enc.drop(columns=['PassengerId'])

'''
Now it looks like enough preprocessing has been done to the training data.
Below, all the same preprocessing will be done to the testing set.
'''
#%%
# Fill the NaN values with 0
test = test.fillna(0)

# Create the X_test dataframe
X_test = test

# Going to drop the name columns due to little value
X_test= test.drop(columns=['Name'])

'''
The cabin variable is broken into three distinct parts:
    Deck Number
    Room Number
    Side of ship
So break down the cabin variable into these three parts
'''
X_test[['Deck Number', 'Room Number', 'Ship Side']] = test['Cabin'].str.split('/', expand=True)
# Now to drop the entire cabin feature
X_test = X_test.drop(columns=['Cabin'])

# Separate Group Size from Id then drop Id
X_test[['Id','Group Size']] = X_test['PassengerId'].str.split('_',expand=True)
X_test = X_test.drop(columns=['Id'])
X_test['Group Size'] = pd.to_numeric(X_test['Group Size'])


test_categorical = X_test.drop(columns = ['PassengerId','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Room Number'])
# Also need to drop those columns from X_test
X_test = X_test.drop(columns=['HomePlanet','CryoSleep','Destination','VIP','Deck Number','Ship Side'])

# Change the room number from an object to a numeric
X_test['Room Number'] = pd.to_numeric(X_test['Room Number'])

# Prefrom StandardScaler on the numeric variables
X_test[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck', 'Group Size']] = StandardScaler().fit_transform(X_test[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck', 'Group Size']])

# Perform one-hot encoding on the categorical variables then concatenate the two dataframes
test_cat_encoded = pd.get_dummies(test_categorical)
test_cat_encoded = test_cat_encoded.astype(float)
test_cat_encoded.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test_enc = pd.concat([X_test,test_cat_encoded], axis=1)

# Get rid of the PassengerId column
X_test_enc = X_test_enc.drop(columns=['PassengerId'])

#%% 30% Testing split
'''
Now to actually train a model. Unfortunately, to figure out how well it works,
I will have to submit it to the competition. Luckily that is not too hard.
'''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

'''
The models will start with the simpler linear and logistic regression models
and increase in complexity further down.
'''
# Somehow a few NaN values are in X_train_enc so those rows are being deleted
X_train_enc = X_train_enc.fillna(0)


# Split up data from the X_train_enc dataset
X_train_train,X_train_test,y_train_train,y_train_test = train_test_split(X_train_enc,y_train,test_size=0.3,random_state=10)

# Decision tree classifier
# Need to see the best max depth

acc_dt_list = []
l = []
i = 1
while i < 51:
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(X_train_train,y_train_train)
    y_pred_dt_l = dt.predict(X_train_test)
    acc_l = accuracy_score(y_train_test,y_pred_dt_l)
    acc_dt_list.append(acc_l)
    l.append(i)
    i += 1
acc_dt_l = np.array(acc_dt_list)
n = np.array(l)
plt.plot(n,acc_dt_l)
plt.show()

# It looks like the optimal depth is 8 according to the graph
dt = DecisionTreeClassifier(max_depth=8, random_state=10)
dt.fit(X_train_train,y_train_train)
y_pred_dt = dt.predict(X_train_test)
acc_dt = accuracy_score(y_train_test,y_pred_dt)

fp_dt, tp_dt, dt_thresholds = roc_curve(y_train_test, y_pred_dt)
roc_auc_dt = auc(fp_dt, tp_dt)

# Random Forest Classification

# Going to look at what the best number would be for max_depth
'''
acc_rf_list = []
l = []
i = 18
while i <30:
    rf = RandomForestClassifier(max_depth=i, random_state=10)
    rf = rf.fit(X_train_train, y_train_train)
    y_pred_rf_l = rf.predict(X_train_test)
    acc_l = accuracy_score(y_train_test, y_pred_rf_l)
    acc_rf_list.append(acc_l)
    l.append(i)
    i += 1
acc_rf_l = np.array(acc_rf_list)
n = np.array(l)
plt.plot(n,acc_rf_l)
plt.show()
'''
# 25 appears to be the optimal amount

rf = RandomForestClassifier(max_depth=25)
rf.fit(X_train_train, y_train_train)
y_pred_rf = rf.predict(X_train_test)
acc_rf = accuracy_score(y_train_test, y_pred_rf)
# Calculating roc and auc for random forest
fp_rf, tp_rf, thresholds_rf = roc_curve(y_train_test,y_pred_rf)
roc_auc_rf = auc(fp_rf,tp_rf)

# K-Nearest Neighbors Classification
'''
Code used to find optimal n_neighbors number
acc_knn_list = []
l = []
i = 1
while i < 51:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train_train, y_train_train)
    y_pred_knn_forlist = knn.predict(X_train_test)
    acc_for_list = accuracy_score(y_train_test, y_pred_knn_forlist)
    acc_knn_list.append(acc_for_list)
    l.append(i)
    i += 1
knn_acc_list = np.array(acc_knn_list)
n = np.array(l)
plt.plot(n,knn_acc_list)
plt.show()

Looks like 7 or 8 is optimal number
Taking a closer look it was found to be 8
'''

knn = KNeighborsClassifier(n_neighbors=8)
knn = knn.fit(X_train_train, y_train_train)
y_pred_knn = knn.predict(X_train_test)
acc_knn = accuracy_score(y_train_test, y_pred_knn)

fp_knn, tp_knn, thresholds_knn = roc_curve(y_train_test, y_pred_knn)
roc_auc_knn = auc(fp_knn,tp_knn)

# Neural Net Classification

model = Sequential()
model.add(Dense(12,input_dim=30,activation='softmax'))
model.add(Dense(8,activation='softmax'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_train,y_train_train,epochs=20)

_, acc_nn = model.evaluate(X_train_test,y_train_test)

# Not entirely sure how to optimize this neural net or find auc
'''
I have played around with this model a little bit and each time the accuracy
is between .74-.78. Not knowing how to optimize these neural nets, this will
remain as the accuracy for this classifier
'''

# MLP Classification

mlp = MLPClassifier(random_state=10,max_iter=10).fit(X_train_train,y_train_train)
y_pred_mlp = mlp.predict(X_train_test)
acc_mlp = accuracy_score(y_train_test,y_pred_mlp)

fp_mlp,tp_mlp,thresholds_mlp = roc_curve(y_train_test,y_pred_mlp)
roc_auc_mlp = auc(fp_mlp,tp_mlp)

'''
The MLP Classifier found an accuracy pretty close to the keras neural net above
'''

tts_thirty_percent_acc = {'Decision Tree':acc_dt, 'Random Forest':acc_rf, 'K Neighbors':acc_knn,'Sequential NN':acc_nn,'MLP Classifier':acc_mlp}



#%% Changing up the train test splits

# Somehow a few NaN values are in X_train_enc so those rows are being deleted
#X_train_enc = X_train_enc.fillna(0)

# 90% training with 10% testing
X_train_90, X_test_90, y_train_90, y_test_90 = train_test_split(X_train_enc,y_train,test_size=0.1,random_state=10)

# Decision tree classifier
# Need to see the best max depth
'''
acc_dt_list = []
l = []
i = 10
while i < 15:
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(X_train_90,y_train_90)
    y_pred_dt_l = dt.predict(X_test_90)
    acc_l = accuracy_score(y_test_90,y_pred_dt_l)
    acc_dt_list.append(acc_l)
    l.append(i)
    i += 1
acc_dt_l = np.array(acc_dt_list)
n = np.array(l)
plt.plot(n,acc_dt_l)
plt.show()
'''
# It looks like the optimal depth is 12
dt = DecisionTreeClassifier(max_depth=20, random_state=10)
dt.fit(X_train_90,y_train_90)
y_pred_dt = dt.predict(X_test_90)
acc_dt_90 = accuracy_score(y_test_90,y_pred_dt)

fp_dt, tp_dt, dt_thresholds = roc_curve(y_test_90, y_pred_dt)
roc_auc_dt_90 = auc(fp_dt, tp_dt)

# Random Forest Classification

#Going to look at what the best number would be for max_depth
'''
acc_rf_list = []
l = []
i = 10
while i <19:
    rf = RandomForestClassifier(max_depth=i, random_state=10)
    rf = rf.fit(X_train_90, y_train_90)
    y_pred_rf_l = rf.predict(X_test_90)
    acc_l = accuracy_score(y_test_90, y_pred_rf_l)
    acc_rf_list.append(acc_l)
    l.append(i)
    i += 1
acc_rf_l = np.array(acc_rf_list)
n = np.array(l)
plt.plot(n,acc_rf_l)
plt.show()

15 appears to be the optimal amount
'''
rf = RandomForestClassifier(max_depth=15)
rf.fit(X_train_90, y_train_90)
y_pred_rf = rf.predict(X_test_90)
acc_rf_90 = accuracy_score(y_test_90, y_pred_rf)
# Calculating roc and auc for random forest
fp_rf, tp_rf, thresholds_rf = roc_curve(y_test_90,y_pred_rf)
roc_auc_rf_90 = auc(fp_rf,tp_rf)

# K-Nearest Neighbors Classification

#Code used to find optimal n_neighbors number
'''
acc_knn_list = []
l = []
i = 1
while i < 51:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train_90, y_train_90)
    y_pred_knn_forlist = knn.predict(X_test_90)
    acc_for_list = accuracy_score(y_test_90, y_pred_knn_forlist)
    acc_knn_list.append(acc_for_list)
    l.append(i)
    i += 1
knn_acc_list = np.array(acc_knn_list)
n = np.array(l)
plt.plot(n,knn_acc_list)
plt.show()
'''
#Looks like 8 is optimal number


knn = KNeighborsClassifier(n_neighbors=8)
knn = knn.fit(X_train_90, y_train_90)
y_pred_knn = knn.predict(X_test_90)
acc_knn_90 = accuracy_score(y_test_90, y_pred_knn)

fp_knn, tp_knn, thresholds_knn = roc_curve(y_test_90, y_pred_knn)
roc_auc_knn_90 = auc(fp_knn,tp_knn)

# Neural Net Classification

model = Sequential()
model.add(Dense(12,input_dim=30,activation='softmax'))
model.add(Dense(8,activation='softmax'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_90,y_train_90,epochs=20)

_, acc_nn_90 = model.evaluate(X_test_90,y_test_90)

# Not entirely sure how to optimize this neural net

# MLP Classification

mlp = MLPClassifier(random_state=10,max_iter=10).fit(X_train_90,y_train_90)
y_pred_mlp = mlp.predict(X_test_90)
acc_mlp_90 = accuracy_score(y_test_90,y_pred_mlp)

fp_mlp,tp_mlp,thresholds_mlp = roc_curve(y_test_90,y_pred_mlp)
roc_auc_mlp_90 = auc(fp_mlp,tp_mlp)

# Create dictionary with all accuracy results
tts_ten_percent_acc = {'Decision Tree':acc_dt_90, 'Random Forest':acc_rf_90, 'K Neighbors':acc_knn_90,'Sequential NN':acc_nn_90,'MLP Classifier':acc_mlp_90}


#%% Fifty Fifty train test split

# Somehow a few NaN values are in X_train_enc so those rows are being deleted
#X_train_enc = X_train_enc.fillna(0)

# 50% training with 50% testing
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X_train_enc,y_train,test_size=0.5,random_state=10)

# Decision tree classifier
# Need to see the best max depth
'''
acc_dt_list = []
l = []
i = 1
while i < 10:
    dt = DecisionTreeClassifier(max_depth=i)
    dt.fit(X_train_50,y_train_50)
    y_pred_dt_l = dt.predict(X_test_50)
    acc_l = accuracy_score(y_test_50,y_pred_dt_l)
    acc_dt_list.append(acc_l)
    l.append(i)
    i += 1
acc_dt_l = np.array(acc_dt_list)
n = np.array(l)
plt.plot(n,acc_dt_l)
plt.show()
'''
# It looks like the optimal depth is 8
dt = DecisionTreeClassifier(max_depth=8, random_state=10)
dt.fit(X_train_50,y_train_50)
y_pred_dt = dt.predict(X_test_50)
acc_dt_50 = accuracy_score(y_test_50,y_pred_dt)

fp_dt, tp_dt, dt_thresholds = roc_curve(y_test_50, y_pred_dt)
roc_auc_dt_50 = auc(fp_dt, tp_dt)

# Random Forest Classification

#Going to look at what the best number would be for max_depth
'''
acc_rf_list = []
l = []
i = 20
while i <30:
    rf = RandomForestClassifier(max_depth=i, random_state=10)
    rf = rf.fit(X_train_50, y_train_50)
    y_pred_rf_l = rf.predict(X_test_50)
    acc_l = accuracy_score(y_test_50, y_pred_rf_l)
    acc_rf_list.append(acc_l)
    l.append(i)
    i += 1
acc_rf_l = np.array(acc_rf_list)
n = np.array(l)
plt.plot(n,acc_rf_l)
plt.show()
'''
# The accuracy appears to taper off around 22

rf = RandomForestClassifier(max_depth=22)
rf.fit(X_train_50, y_train_50)
y_pred_rf = rf.predict(X_test_50)
acc_rf_50 = accuracy_score(y_test_50, y_pred_rf)
# Calculating roc and auc for random forest
fp_rf, tp_rf, thresholds_rf = roc_curve(y_test_50,y_pred_rf)
roc_auc_rf_50 = auc(fp_rf,tp_rf)

# K-Nearest Neighbors Classification

# Code used to find optimal n_neighbors number
'''
acc_knn_list = []
l = []
i = 1
while i < 5:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train_50, y_train_50)
    y_pred_knn_forlist = knn.predict(X_test_50)
    acc_for_list = accuracy_score(y_test_50, y_pred_knn_forlist)
    acc_knn_list.append(acc_for_list)
    l.append(i)
    i += 1
knn_acc_list = np.array(acc_knn_list)
n = np.array(l)
plt.plot(n,knn_acc_list)
plt.show()
'''
# Looks like 1 is optimal number


knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train_50, y_train_50)
y_pred_knn = knn.predict(X_test_50)
acc_knn_50 = accuracy_score(y_test_50, y_pred_knn)

fp_knn, tp_knn, thresholds_knn = roc_curve(y_test_50, y_pred_knn)
roc_auc_knn_50 = auc(fp_knn,tp_knn)

# Neural Net Classification

model = Sequential()
model.add(Dense(12,input_dim=30,activation='softmax'))
model.add(Dense(8,activation='softmax'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_50,y_train_50,epochs=20)

_, acc_nn_50 = model.evaluate(X_test_50,y_test_50)

# Not entirely sure how to optimize this neural net

# MLP Classification

mlp = MLPClassifier(random_state=10,max_iter=10).fit(X_train_50,y_train_50)
y_pred_mlp = mlp.predict(X_test_50)
acc_mlp_50 = accuracy_score(y_test_50,y_pred_mlp)

fp_mlp,tp_mlp,thresholds_mlp = roc_curve(y_test_50,y_pred_mlp)
roc_auc_mlp_50 = auc(fp_mlp,tp_mlp)

# Create dictionary with all accuracy results
tts_fifty_percent_acc = {'Decision Tree':acc_dt_50, 'Random Forest':acc_rf_50, 'K Neighbors':acc_knn_50,'Sequential NN':acc_nn_50,'MLP Classifier':acc_mlp_50}


