# Importing Libraries
# Importing dataset from library
from sklearn.datasets import load_iris
# Importing perceptron model
from sklearn.linear_model import Perceptron
# Importing a performance metric
from sklearn.metrics import accuracy_score

# Loading dataset
iris = load_iris()

# Printing dataset
print(iris)

# Independent and Dependent Variables
x = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(int)

# Applying the model
ptron = Perceptron(random_state=42)

# Fitting values to model
ptron.fit(x, y)

# Predicting using model
y_pred = ptron.predict(x)

# Printing prediction results
print(y_pred)

# Accuracy of model
print(f'Accuracy Score : {accuracy_score(y, y_pred)}')

# Coefficient values from model
print(f'Coefficient Values : {ptron.coef_}')

# Intercept value from model
print(f'Intercept Value : {ptron.intercept_}')
