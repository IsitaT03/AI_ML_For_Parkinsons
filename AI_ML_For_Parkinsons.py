#!/usr/bin/env python
# coding: utf-8

##############################################
#
# This program experiments with the performance of AI and ML techniques for Predicting Parkinsons' Disease.
# This code is designed to test multiple algorithms using the same pre-processed data.
# 
# Written by Isita Talukdar
#
##############################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv

##Control what tests to run
RUN_ANN =False
RUN_ANNCROSSVAL = False
RUN_ANNGRIDSEARCH = False
RUN_ANNMANUALGS =True
RUN_RANDFOREST =False
RUN_KNN = False
RUN_BAYES = False


#===============================#
# Data Preparation
#===============================#

#Import data from CSV file
df = pd.read_csv("FoxInsight.csv")
#df.head(10)

#merges the rows with same id and takes most recent valid values for each column
df = df.groupby('fox_insight_id').last()

#All features with range 1-5 fill with default 1(No Diagnosed Problem)
for column in df[['MoveSpeech', 'MoveSaliva','MoveWrite','MoveTremor','MoveWalk']]: 
    df[column] = df[column].fillna(1)
#All features with 0 or 1 or 2 fill with default 0(This family member does not have PD) or 2(There is no Family History aof a disease)
for column in df[['FamParkinsonMoth', 'FamParkinsonFath', 'FamParkinsonMatGrMoth','FamParkinsonMatGrFath','FamParkinsonPatGrMoth', 'FamParkinsonPatGrFath','FamParkinsonOth', 'NonMoveForget','NonMoveConcent','NonMoveDizzy']]:
    df[column] = df[column].fillna(0)    
for column in df[['FamParkinsonHx', 'FamAlzheimerHx', 'FamALSHx','FamAutismHx','FamDystoniaHx', 'FamEpilepsyHx','FamMSHx', 'FamStrokeHx','FamBipolarHx','FamDepressionHx','FamAnxietyHx','FamSuicideHx']]:
    df[column] = df[column].fillna(2)

#Dropped entries with no diagnosis or sex information in the database   
df = df.dropna(subset=['CurrPDDiag'])
df1 = df.dropna(subset=['Sex'])
#print("Amnt after removing no diagnosis and sex:",len(df))

#Create Pandas Dataframe of Selected Features
X_Phys = df1.iloc[:, [df1.columns.get_loc("MoveSpeech"),df1.columns.get_loc("MoveSaliva"), df1.columns.get_loc("MoveTremor"), 
           df1.columns.get_loc("MoveWalk"), df1.columns.get_loc("NonMoveForget"), df1.columns.get_loc("NonMoveConcent"),df1.columns.get_loc("NonMoveDizzy")]].values
X_Fam = df1.iloc[:, [df1.columns.get_loc("FamParkinsonMoth"),df1.columns.get_loc("FamParkinsonFath"),df1.columns.get_loc("FamParkinsonMatGrMoth"),df1.columns.get_loc("FamParkinsonMatGrFath"), df1.columns.get_loc("FamParkinsonPatGrMoth"), 
            df1.columns.get_loc("FamParkinsonPatGrFath"), df1.columns.get_loc("FamParkinsonOth")]]
X_Hx = df1.iloc[:, [df1.columns.get_loc("FamParkinsonHx"), df1.columns.get_loc("FamAlzheimerHx"), df1.columns.get_loc("FamALSHx"), 
            df1.columns.get_loc("FamAutismHx"), df1.columns.get_loc("FamDystoniaHx"), df1.columns.get_loc("FamEpilepsyHx"), df1.columns.get_loc("FamMSHx"), df1.columns.get_loc("FamStrokeHx"), 
            df1.columns.get_loc("FamBipolarHx"), df1.columns.get_loc("FamDepressionHx"), df1.columns.get_loc("FamAnxietyHx"), df1.columns.get_loc("FamSuicideHx")]]
y = df1.iloc[:,df1.columns.get_loc("CurrPDDiag")].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Apply OneHotEncoder on categorical feature columns
ohe = OneHotEncoder( sparse = False)
X_Hx_ohe = ohe.fit_transform(X_Hx)

#Reassemble full, pre-processed data
X = np.concatenate((X_Fam, X_Hx_ohe, X_Phys), axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Confirm there are both 0's and 1's in the arrays
from collections import Counter
trainDist = Counter(y_train)
testDist = Counter(y_test)

#=====================================#
#=====================================#
# Create and Evaluate Different Models
#=====================================#
#=====================================#

#=====================================#
# Create and Evaluate ANN
#=====================================#

from keras.callbacks import TensorBoard
import time

#Make Log Directory for the Tensorbaord files
NAME = "ANN_Only_Fam_Hx_Phys_45_20_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#Single Config Test(One training run)
if(RUN_ANN):
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    #print("Running ANN")
    #print('Setting up classifier for Phys')

    #Constructing ANN 
    classifier = Sequential()
    classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu', input_dim = 62))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    print("Running Initial Training and Fitting")
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 10, callbacks = [tensorboard])
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    #Using Confusion Matrix to Analyze Accuracy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("End ANN")

#Training With Cross Validation
if(RUN_ANNCROSSVAL):
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    print("Running K-Fold Cross Validation")
    def build_classifier_Phys():
        classifier = Sequential()
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier

    classifier = KerasClassifier(build_fn = build_classifier_Phys, batch_size = 10, nb_epoch = 10)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("Mean: ", mean)
    print("Variance: ", variance)
    print("End K-Fold Cross Validation")

#Training With Built-In Grid Search
if(RUN_ANNGRIDSEARCH):
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    print("Running Grid Search")

    #Create constructor function
    def build_classifier(optimizer, activ):
        classifier = Sequential()
        classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = activ, input_dim = 62))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = activ))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = activ))
        classifier.add(Dropout(p = 0.1))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier

    classifier = KerasClassifier(build_fn = build_classifier)
    print("ANN Start Grid Search")
    parameters = {'batch_size':[10, 25, 50],
                  'nb_epoch': [100, 500],
                  'optimizer':['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier, 
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    best_accuracy  =grid_search.best_score_
    print("best parameters: ", best_parameters)
    print("best accuracy: ", best_accuracy)
    print("ANN End Grid Search")

#Training With Manual Grid Search to create Individul Tensorboard Graph
if(RUN_ANNMANUALGS):
    import tensorflow as tf
    import datetime, os
    from tensorboard.plugins.hparams import api as hp
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    
    #Create constructor function for use in Grid Search Loop
    def train_test_model(
        ep_num = 10,
        batch_sz = 4,
        hidden_layer_cnt = 2,
        hidden_layer_size_factor = 2,
        optimiz = 'adam',
        timeStamp = 0
        ):
        classifier = Sequential()
        #Compute hidden layer size based on input size
        hidden_layer_size = int(hidden_layer_size_factor*62)
        #Add Input Layer and 1st Hidden Layer
        classifier.add(Dense(output_dim = hidden_layer_size, init = 'uniform', activation = 'relu', input_dim = 62))
        #N additional Hidden Layers
        for hidden_layer_index in range(0, hidden_layer_cnt):
            classifier.add(Dense(output_dim = hidden_layer_size, init = 'uniform', activation = 'relu'))
        #add Output Layer
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        #Compiling ANN
        classifier.compile(optimizer = optimiz ,loss = 'binary_crossentropy', metrics = ['accuracy'])
        #Name order ANN_ManGrid_EpochNumber_BatchSize_HiddenLayerCount_HiddenLayerSizeFactor_Optimizer
        logdir_name = "logs_"+str(timeStamp)
        NAME = "ANN_ManGrid_EP"+str(ep_num)+"_BS"+str(batch_sz)+"_HLC"+str(hidden_layer_cnt)+"_HLSF"+str(hidden_layer_size_factor)+"_OP"+str(optimiz)
        log_dir=logdir_name+'/{}'.format(NAME)
        print("logging:    ", log_dir)
        tensorboard = TensorBoard(log_dir)
    
        #Fit ANN into Training Set
        classifier.fit(X_train, y_train, batch_size = batch_sz, nb_epoch = ep_num, callbacks=[tensorboard])
        
    #Using For Loop to Manual Grid Search    
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    opti_list =  ['adam', 'rmsprop']
    for batch_size in range(10, 30, 5):
        for epoch_index in range(10,100 , 10):
            for hid_cnt in range(2, 5, 1):
                for hid_sz_fact in range(1, 3, 1):
                    for optimizer in opti_list:
                        print(" Test with Ep: ", epoch_index, " BatSz: ",  batch_size, " HidLayNum: ", hid_cnt, " SzFactor: ", hid_sz_fact, " Optimizer: ", optimizer)
                        train_test_model(
                                    ep_num = epoch_index,
                                    batch_sz = batch_size,
                                    hidden_layer_cnt = hid_cnt,
                                    hidden_layer_size_factor = hid_sz_fact,
                                    optimiz = optimizer,
                                    timeStamp = time_stamp
                                   )


#=====================================#
# Create and Evaluate Random Forest
#=====================================#
if(RUN_RANDFOREST):
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    #Accuracy Imports
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    
    # Instantiate model 
    rf = RandomForestClassifier()
    # Train the model on training data
    rf.fit(X_train, y_train);
    
    # Use the forest's predict method on the test data
    y_pred = rf.predict(X_test)
    
    # Train and Test Accuracy
    print("Train Accuracy :: ", accuracy_score(y_train, rf.predict(X_train)))
    print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    print(random_grid)
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    print("Best Params: ",rf_random.best_params_)
    
    best_random = rf_random.best_estimator_
    y_pred = best_random.predict(X_test)
    print("BP Train Accuracy :: ", accuracy_score(y_train, best_random.predict(X_train)))
    print("BP Test Accuracy  :: ", accuracy_score(y_test, y_pred))
    
    
#=====================================#
# Create and Evaluate KNN
#=====================================#
if(RUN_KNN):
    #Import knearest neighbors Classifier model
    from sklearn.neighbors import KNeighborsClassifier

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)

    ##Tuning
    from matplotlib      import pyplot as plt
    from IPython.display import display

    print("KNN: Start Gridsearch")
    from sklearn.model_selection import GridSearchCV
    parameters = {'n_neighbors':[1,3,5],}
    grid_search = GridSearchCV(estimator = knn, 
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    best_accuracy  =grid_search.best_score_
    print("best parameters: ", best_parameters)
    print("best accuracy: ", best_accuracy)
    print(grid_search.cv_results_)
    print("KNN: End Gridsearch")
    
#=====================================#
# Create and Evaluate NAIVE BAYES: Gaussian and Multinomial
#=====================================#
if(RUN_BAYES):
    from sklearn.naive_bayes import GaussianNB
    
    #Create Gaussian Classifier
    gnb = GaussianNB()
   
    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    
    from sklearn import metrics
    print("Gaussian NB Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
     #Create Multinomial Classifier
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()

    #Train the model using the training sets
    mnb.fit(X_train_orig, y_train)

    #Predict the response for test dataset
    y_pred = mnb.predict(X_test_orig)
    
    from sklearn import metrics
    print("Multinomial NB Accuracy:",metrics.accuracy_score(y_test, y_pred))





