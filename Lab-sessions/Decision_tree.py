import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


data = load_iris()
# defining a pandas dataframe 
# df2=pd.read_csv("D:/UCSC/Academic/Year 04/Data Analytics/mnist_train.csv")
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
df['target'] = data.target

# split data 
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

clf = DecisionTreeClassifier(max_depth = 3, random_state = 0)
# train model
clf.fit(X_train, Y_train)

# predicting labels
y_pred=clf.predict(X_test)

# model accuracy
print("Accuracy:",accuracy_score(Y_test, y_pred))

# accuracy report
print("Accuracy:",classification_report(Y_test, y_pred))


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

# Setting dpi = 300 to make image clearer than default
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,3), dpi=300)

tree.plot_tree(clf,
           feature_names = fn, 
           class_names=cn,
           filled = True)

fig.savefig('tree_iris.png')
plt.show()
