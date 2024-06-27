import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Criando dados de treino e teste
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Apply the LOF algorithm
lof = LocalOutlierFactor(n_neighbors=5, novelty=True)

# Fit the model to the data (novelty detection) 
lof.fit(X_train)

# Obtendo as previsões do modelo no conjunto de teste
y_pred_test = lof.predict(X_test)

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
plt.title('Anomly by LOF',fontsize=16)  
plt.savefig("lof/noveltyIris.png")
