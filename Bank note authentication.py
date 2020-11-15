import os
os.getcwd()  
os.chdir('D:\Data Science\Project\Project Bank note')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


data=pd.read_csv("D:\\Data Science\\Project\\Project Bank note\\BankNote_Authentication.csv")

# Summarize Data

# Descriptive statistics
# shape
print(data.shape)
#head
print(data.head(20))
#tail
print(data.tail(20))
data.describe

data.isnull().sum()

# Data visualizations
#Hist#
data.hist()
plt.show()

#Output barplot#
sns.set_style("darkgrid")
lf=data["class"].value_counts()
temp=sns.barplot(lf.index,lf.values,alpha=0.8)
plt.title("Output")
plt.show()

#Corelation
corelation=data.drop(["class"],axis=1).corr()
corr_heatmap=sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# Prepare Data

# Split-out validation dataset

X=data.iloc[:,0:4]
y=data.iloc[:,4]

validation_size=0.3
seed_size=7
scoring="accuracy"
num_folds = 10
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=validation_size,random_state=seed_size)

# Evaluate Algorithms

models=[]
models.append(("LR",LogisticRegression()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(('SVM', SVC()))

names=[]
results=[]


for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed_size)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Standardize the dataset

pipelines=[]
pipelines.append(("ScaledLR",Pipeline([("Scaler",StandardScaler()),("LR",LogisticRegression())])))
pipelines.append(("ScaledDT",Pipeline([("Scaler",StandardScaler()),("DT",DecisionTreeClassifier())])))
pipelines.append(("ScaledNB",Pipeline([("Scaler",StandardScaler()),("NB",GaussianNB())])))
pipelines.append(("ScalerKNN",Pipeline([("Scaler",StandardScaler()),("KNN",KNeighborsClassifier())])))
pipelines.append(("ScalerSVM",Pipeline([("Scaler",StandardScaler()),("SVM",SVC())])))
results = []
names = []
for name,model in pipelines:
    kfold=KFold(n_splits=num_folds,random_state=seed_size)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)
    
fig=plt.figure()    
fig.suptitle("Standardized Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



#By using ensembles#

ensembles=[]
ensembles.append(("AB",AdaBoostClassifier()))
ensembles.append(("GB",GradientBoostingClassifier()))
ensembles.append(("RF",RandomForestClassifier()))
ensembles.append(("ET",ExtraTreesClassifier()))
results=[]
names=[]

for name,model in ensembles:
    kfold=KFold(n_splits=num_folds,random_state=seed_size)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)
    
fig=plt.figure()    
fig.suptitle("Ensemble Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#Finalizing the model#

classifier=GaussianNB()
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
np.mean(y_test==pred)
#Accuracy=82.76 which is neither overfitting nor underfitting#



import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close


#TEsting our model#

nb.predict([[-1.6,1.09,-0.35,-0.59]])
#array([1], dtype=int64)

