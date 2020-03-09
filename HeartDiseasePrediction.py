# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:25:00 2019

@author: Salman
"""

#1 IMPORT ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.cm import rainbow
#matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#from sklearn.tree import DecisionTreeClassifier

df =pd.read_csv('heart.csv')

#df.info()

#df.describe()

#########################FEATURE SELECTION#####################################

#get correlations of each features in dataset
#corrmat = df.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
##plot heat map
#g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#df.hist()

#CHECK WHETHER THE DATA IS BALANCED OR NOT
#sns.set_style('whitegrid')
#sns.countplot(x='target',data=df,palette='RdBu_r')
#As you can see that number of 0s and 1s is pretty much equal

#IN THE DATASET IT CAN BE OBSERVED THAT SOME OF THE COLUMNS HAVE CATEGORICAL DATA 
#WHICH NEEDS TO CONVERTED INTO DUMMY VARIABLES AND SCALE ALL THE VARIABLES

#########################FEATURE ENGINEERING###################################
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#AS YOU CAN SEE IN THE DATASET SOME OF THE COLUMNS' VALUE VARY OR ARE DEFINED IN DIFFERENT UNITS
#SO STANDARDIZATION IS REQUIRED

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#VERIFY BOTH THE ABOVE STEPS BY USING
#dataset.head()

#DEFINE INDEPENDENT AND DEPENDENT FEATURES FROM DATASET
y=dataset['target']
X=dataset.drop(['target'],axis=1)

#########################MODEL PROCESSING#####################################

#FIND THE CROSS VALIDATION SCORE OF KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
    
#PLOT THE RESULT AND LOOK FOR THE HIGHEST N_NEIGHBOR VALUE (12 in this case)
#plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
#for i in range(1,21):
#    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
#plt.xticks([i for i in range(1, 21)])
#plt.xlabel('Number of Neighbors (K)')
#plt.ylabel('Scores')
#plt.title('K Neighbors Classifier scores for different K values')    

#COMPUTE THE CROSS VALIDATION SCORE WITH k=12 AS IT SHOWS THE HIGHEST ACCURACY
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score_knn=cross_val_score(knn_classifier,X,y,cv=10)

#NOW FIND THE MEAN VALUE OF 10 CASES 
print('ACCURACY OF KNN: ' + str(score_knn.mean()))

#FIND THE CROSS VALIDATION SCORE OF RANDOMFOREST
from sklearn.ensemble import RandomForestClassifier
RF_scores = []
for rf in range(1,21):
    randomforest_classifier = RandomForestClassifier(n_estimators = rf)
    score=cross_val_score(randomforest_classifier,X,y,cv=10)
    RF_scores.append(score.mean())

#PLOT THE RESULT AND IT CAN BE SEEN THAT WITH N_ESTIMATOR = 12
#plt.plot([rf for rf in range(1, 21)], RF_scores, color = 'red')
#for i in range(1,21):
#    plt.text(i, RF_scores[i-1], (i, RF_scores[i-1]))
#plt.xticks([i for i in range(1, 21)])
#plt.xlabel('Number of Estimators (rf)')
#plt.ylabel('Scores')
#plt.title('RandomForest Classifier scores for different n_estimator values') 

#PLOT THE RESULT AND LOOK FOR THE HIGHEST N_ESTIMATOR VALUE (19 in this case)
randomforest_classifier = RandomForestClassifier(n_estimators = 19)
score_rf=cross_val_score(randomforest_classifier,X,y,cv=10) 

#NOW FIND THE MEAN VALUE OF 10 CASES 
print('ACCURACY OF RANDOM FOREST: ' + str(score_rf.mean()))

#FIND THE CROSS VALIDATION SCORE OF DECISIONTREE
from sklearn.tree import DecisionTreeClassifier
dt_scores = []
for dt in range(1,21):
    decisiontree_classifier = DecisionTreeClassifier(random_state = dt)
    score=cross_val_score(decisiontree_classifier,X,y,cv=10)
    dt_scores.append(score.mean())

#PLOT THE RESULT AND IT CAN BE SEEN THAT WITH N_ESTIMATOR = 12
#plt.plot([dt for dt in range(1, 21)], dt_scores, color = 'red')
#for i in range(1,21):
#    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
#plt.xticks([i for i in range(1, 21)])
#plt.xlabel('Number of Random State (DT)')
#plt.ylabel('Scores')
#plt.title('Decision Tree Classifier scores for different random_state values') 

#PLOT THE RESULT AND LOOK FOR THE HIGHEST random_state value (4 in this case)
decisiontree_classifier = DecisionTreeClassifier(random_state = 4)
score_dt=cross_val_score(decisiontree_classifier,X,y,cv=10)

#NOW FIND THE MEAN VALUE OF 10 CASES 
print('ACCURACY OF DECISION TREE: ' + str(score_dt.mean()))
