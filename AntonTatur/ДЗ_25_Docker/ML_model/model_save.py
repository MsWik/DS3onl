from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first feature.
y = iris.target

model = DecisionTreeClassifier()
model.fit (X, y)

root = str(os.path.dirname(os.path.realpath(__file__)))
my_path = root + "\Model.pkl"
with open(my_path, 'wb') as file:  
    pickle.dump(model, file)

