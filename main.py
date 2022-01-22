# https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv - DATASET

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# Loading data
dataset = pandas.read_csv("titanic.csv")

# Getting rid of data that wont be used
dataset.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
# Some functions requre floats, so we need to change the dataset a bit:
d = {'male': 0, 'female': 1}
dataset['Sex'] = dataset['Sex'].map(d)

# We dont know every person's age so we have to fill blank values with mean.
dataset.Age = dataset.Age.fillna(dataset.Age.mean())

# Dividing the values into parameters and the result
inputs = dataset.drop('Survived', axis='columns')
target = dataset.Survived

# Creating the Tree
dtree = DecisionTreeClassifier(max_depth=4)
dtree = dtree.fit(inputs, target)

# Functions necessary for exporting tree to png
data = tree.export_graphviz(dtree, out_file=None, feature_names=dataset.columns[1:5],
                            class_names=['Died', 'Survive'], rounded=True, filled=True)

graph = pydotplus.graph_from_dot_data(data)
graph.write_png('titanic.png')

img = pltimg.imread('titanic.png')
imgplot = plt.imshow(img)

