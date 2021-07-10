from id3 import Id3Estimator, export_graphviz,export_text
from graphviz import Source
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

clf = Id3Estimator()
clf.fit(X_train, Y_train)
y_pred=clf.predict(X_test)

# model accuracy
print("Accuracy:",accuracy_score(Y_test, y_pred))

# accuracy report
print("Accuracy:",classification_report(Y_test, y_pred))


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

# Setting dpi = 300 to make image clearer than default
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,3), dpi=300)
export_graphviz(clf.tree_, "out.dot", data.feature_names)
export_text(clf.tree_,data.feature_names)



