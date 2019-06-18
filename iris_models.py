#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

iris_data = pd.read_csv('iris.data', sep=',', header=None)

iris_data.columns = ["sepal_length","sepal_width","petal_length","petal_width","class"]

X = iris_data.loc[:,["sepal_length","sepal_width","petal_length","petal_width"]].values
y = iris_data.loc[:,'class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Train model on:
# 1 - All features
# 2 - No new features

# Models to test
# A - Random Forest
# B - Support Vector Classifier

# Random Forest Classifier

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)

scores = cross_val_score(rf_clf, X_train, y_train, cv=10)
print("Randon Forrest Classifier")
print("All cross validation runs: %s" % (scores))
print("Average: %s" % (scores.mean()))

rf_clf.fit(X_train, y_train)
final_score = rf_clf.score(X_test,y_test)

print("Performance on test data: %s" % (final_score))

# Linear Support Vector Classifier
svc_clf = LinearSVC(tol=1e-5,max_iter=5000)

scores = cross_val_score(svc_clf, X_train, y_train, cv=10)

print("")


print("Linear Support Vector Classifier")
print("All cross validation runs: %s" % (scores))
print("Average: %s" % (scores.mean()))


svc_clf.fit(X_train, y_train)
final_score = svc_clf.score(X_test,y_test)

print("Performance on test data: %s" % (final_score))

# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
# Increasing the number of iterations to 5 resolves this warning