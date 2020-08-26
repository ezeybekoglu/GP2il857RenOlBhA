import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
np.random.seed(1)

df = pd.read_csv('term-deposit-marketing-2020.csv',sep=',')
#print(df.info)

y_value = LabelEncoder()
df['y'] = y_value.fit_transform(df['y'])

job = LabelEncoder()
df['job'] = job.fit_transform(df['job'])

marital = LabelEncoder()
df['marital'] = marital.fit_transform(df['marital'])

education = LabelEncoder()
df['education'] = education.fit_transform(df['education'])

default = LabelEncoder()
df['default'] = default.fit_transform(df['default'])


housing = LabelEncoder()
df['housing'] = housing.fit_transform(df['housing'])

loan = LabelEncoder()
df['loan'] = loan.fit_transform(df['loan'])

contact = LabelEncoder()
df['contact'] = contact.fit_transform(df['contact'])

month = LabelEncoder()
df['month'] = month.fit_transform(df['month'])



print(df.head())
print(df.dtypes)
print(df.columns[df.isnull().any()])

print(df.corr)



y = np.array(df['y'])
X = np.array(df.drop(['y'], 1))




X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
print(Y_test)

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from functools import partial

metrics = {
    "hamming_loss": hamming_loss,
    "subset_accuracy": accuracy_score,
    "macro-f1": partial(f1_score, average="macro"),
    "micro-f1": partial(f1_score, average="micro"),
    "AUC" : auc,
    "weighted-f1": partial(f1_score, average="weighted"),
}


#Fitting Naive_Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
scores = cross_val_score(classifier, X, y, cv=5)
#classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(X_test)

print("GaussianNB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Fitting Decision Tree Algorithm entropy
from sklearn.tree import DecisionTreeClassifier
for n in ["entropy","gini"]:
    classifier = DecisionTreeClassifier(criterion = n, random_state = 0)
    scores = cross_val_score(classifier, X, y, cv=5)
    #classifier.fit(X_train, Y_train)
    #Y_pred = classifier.predict(X_test)
    print("DecisionTreeClassifier "+n+" Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Fitting extra Tree Algorithm entropy
from sklearn.ensemble import ExtraTreesClassifier

for n in ["entropy","gini"]:
    classifier = ExtraTreesClassifier(criterion = n, random_state = 0)
    scores = cross_val_score(classifier, X, y, cv=5)
    #classifier.fit(X_train, Y_train)
    #Y_pred = classifier.predict(X_test)
    print("ExtraTreesClassifier "+n+" Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
scores = cross_val_score(classifier, X, y, cv=5)
#classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(X_test)
print("LogisticRegression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
for n in ["svd","lsqr","eigen"]:
    if n == "svd":    
        classifier = LinearDiscriminantAnalysis(solver = n)
    else:
        classifier = LinearDiscriminantAnalysis(solver = n,shrinkage='auto')
    scores = cross_val_score(classifier, X, y, cv=5)
    #classifier.fit(X_train, Y_train)
    #Y_pred = classifier.predict(X_test)
    print("LinearDiscriminantAnalysis "+n+" Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
classifier = RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)
scores = cross_val_score(classifier, X, y, cv=5)
classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(X_test)
print("RandomForestClassifier Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("*****************predict*************")
#for row in Z:
#    ba.append(classifier.predict(row.reshape(-1,8))[0])

#print("*****************predict*************")
classifier = AdaBoostClassifier()
scores = cross_val_score(classifier, X, y, cv=5)
classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(X_test)
print("AdaBoostClassifier Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()
scores = cross_val_score(classifier, X, y, cv=5)
classifier.fit(X_train, Y_train)
#Y_pred = classifier.predict(Z)
print("QuadraticDiscriminantAnalysis Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("*****************predict*************")
#for row in Z:
#    ba.append(classifier.predict(row.reshape(-1,9))[0])

print("*****************predict*************")
print("QuadraticDiscriminantAnalysis Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print("***********************************")

k_value = 7
from sklearn.neighbors import KNeighborsClassifier
p=1
for n in ["K-NN manhattan_distance","K-NN euclidean_distance"]:
    classifier = KNeighborsClassifier(n_neighbors = k_value, metric = 'minkowski', p = p)
    scores = cross_val_score(classifier, X, y, cv=5)
    #classifier.fit(X_train, Y_train)
    #Y_pred = classifier.predict(X_test)
    print("K-NN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#GaussianNB Accuracy: 0.92 (+/- 0.03)
#DecisionTreeClassifier entropy Accuracy: 0.50 (+/- 0.51)
#DecisionTreeClassifier gini Accuracy: 0.42 (+/- 0.37)
#ExtraTreesClassifier entropy Accuracy: 0.63 (+/- 0.36)
#ExtraTreesClassifier gini Accuracy: 0.65 (+/- 0.33)
#LogisticRegression Accuracy: 0.93 (+/- 0.02)
#LinearDiscriminantAnalysis svd Accuracy: 0.93 (+/- 0.03)
#LinearDiscriminantAnalysis lsqr Accuracy: 0.93 (+/- 0.03)
#LinearDiscriminantAnalysis eigen Accuracy: 0.93 (+/- 0.03)
#RandomForestClassifier Accuracy: 0.90 (+/- 0.06)
#AdaBoostClassifier Accuracy: 0.75 (+/- 0.45)
#QuadraticDiscriminantAnalysis Accuracy: 0.91 (+/- 0.05)
#k=7
#K-NN K-NN manhattan_distance Accuracy: 0.93 (+/- 0.01)
#K-NN K-NN euclidean_distance Accuracy: 0.93 (+/- 0.01)


