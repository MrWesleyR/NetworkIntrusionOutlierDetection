import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo Isolation Forest usando o conjunto de treinamento
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_train)

# Prever se as instâncias no conjunto de teste são outliers ou não
y_pred_test = clf.predict(X_test)

# Contar o número de outliers no conjunto de teste
num_outliers = np.sum(y_pred_test == -1)
print(f"Number of outliers in the test set: {num_outliers}")

# Plotar o resultado destacando os outliers
plt.figure(figsize=(10, 6))

plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], c='blue', label='Normal')
plt.scatter(X_test[y_pred_test == -1, 0], X_test[y_pred_test == -1, 1], c='red', label='Outlier')
plt.xlabel('sepal length (cm)',fontsize=13)
plt.ylabel('sepal width (cm)',fontsize=13)
plt.legend()
plt.title('Anomly by Isolation Forest',fontsize=16)  
plt.savefig("isolation/isolationIris.png")
