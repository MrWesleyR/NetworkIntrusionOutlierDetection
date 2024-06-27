# Import the necessary modules 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn import svm 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Criando dados de treino e teste
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Define the model and set the nu parameter 
model = svm.OneClassSVM(nu=0.05) 

# Fit the model to the data 
model.fit(X_train) 

# Obtendo as previsões do modelo no conjunto de teste
y_pred_test = model.predict(X_test)

# Contar o número de outliers no conjunto de teste
num_outliers = np.sum(y_pred_test == -1)
print(f"Number of outliers in the test set: {num_outliers}")

# Plotar o resultado destacando os outliers
plt.figure(figsize=(10, 6))

plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], c='blue', label='Normal')
plt.scatter(X_test[y_pred_test == -1, 0], X_test[y_pred_test == -1, 1], c='red', label='Outlier')
plt.title('Anomly by One-class Support Vector Machines',fontsize=16) 
plt.xlabel('sepal length (cm)',fontsize=13)
plt.ylabel('sepal width (cm)',fontsize=13)
plt.legend()
plt.savefig("svm/svmIris.png")