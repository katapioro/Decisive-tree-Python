
# Link to data: https://www.kaggle.com/aungpyaeap/tictactoe-endgame-dataset-uci
# Imports for Decision tree itself

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# Loading data

dataset = pandas.read_csv("TicTacToe.csv")

# Further functions require numeric values instead of strings, so we need to make a change
for i in range(1,  10):
    data = {'x': 1, 'o': 2, 'b': 0}
    column_name = "V" + str(i)

    dataset[column_name] = dataset[column_name].map(data)

data = {'positive': 1, 'negative': 0}
dataset['V10'] = dataset['V10'].map(data)

# Separating columns between features and target column:

features = ['V1' ,'V2' ,'V3' ,'V4' ,'V5', 'V6' ,'V7' ,'V8' ,'V9']
X = dataset[features]
y = dataset['V10']

#Create a Decision Tree, save it as an image, and show the image:
dtree = DecisionTreeClassifier(max_depth=4)
dtree = dtree.fit(X, y)

data = tree.export_graphviz(dtree, filled=True, rounded=False,
                special_characters=True,  class_names=['0','1'], feature_names=features)
print(type(data))

graph = pydotplus.graph_from_dot_data(data)
print()
graph.write_png('TreeTacToe.png')

img=pltimg.imread('TreeTacToe.png')
imgplot = plt.imshow(img)
