import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from scipy.io import arff
import pandas as pd

# Load the specific CICIDS2017 dataset
data, meta = arff.loadarff('data/Tuesday.arff')
df = pd.DataFrame(data)
original_shape = df.shape  # shape of the original DataFrame

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with NaN values
df.dropna(axis=0, how='any', inplace=True)

# Calculate the number of removed rows
linhas_removidas = original_shape[0] - df.shape[0]

print(f"Removed {linhas_removidas} rows with NaN values.")

# # Separate features (X) and labels (y)
# X_all = df.drop(columns=['Label']).values
# y_all = df['Label'].values

# Separate normal and outlier data
X_normais = df[df['Label'] == 0].drop(columns=['Label']).values
X_outliers = df[df['Label'] == 1].drop(columns=['Label']).values
y_normais = df[df['Label'] == 0]['Label'].values
y_outliers = df[df['Label'] == 1]['Label'].values
print(f"Tamanho x_normais: {len(X_normais)}")
print(f"Tamanho x_outliers: {len(X_outliers)}")

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_normais, y_normais, test_size=0.3, random_state=42, shuffle=True) 
print(f"Vetor treino: {len(X_train)}")
print(f"Vetor teste: {len(X_test)}")

# Apply the LOF algorithm to the training set
lof = LocalOutlierFactor(n_neighbors=5, novelty=True)

# Fit the model to the data (novelty detection) 
lof.fit(X_train)

y_teste = pd.DataFrame(y_test)

y_teste = y_teste.values

# Concatenate the outliers in the test vector
X_test = np.concatenate((X_test, X_outliers), axis=0)
print(f"Vector test after concat: {len(X_test)}")
y_teste = np.append(y_teste, np.ones(len(X_outliers)))
y_teste = pd.DataFrame(y_teste)

# Predict the model on the test set
y_pred_test = lof.predict(X_test)
pred_test = pd.DataFrame(y_pred_test)

# Count the number of outliers in the test set
num_outliers = np.sum(y_pred_test == -1)
print(f"Number of outliers in the test set: {num_outliers}")

# Function lambda to transform 0 to 1 and 1 to -1
transform_function = lambda x: 1 if x == 0 else -1

# Apply the lambda function to the entire DataFrame
y_teste = y_teste.applymap(transform_function)

# y_teste are the actual labels, y_pred_teste are the model predictions
precision = precision_score(y_teste, y_pred_test)
recall = recall_score(y_teste, y_pred_test)
f1 = f1_score(y_teste, y_pred_test)

def confusion_matrix_scorer(y_teste, y_pred_test):
    cm = confusion_matrix(y_teste, y_pred_test)
    
    # class of interest = outlier
    tn = cm[1,1]
    fp = cm[1,0]
    
    tp = cm[0,0]
    fn = cm[0,1]

    print()
    print("CLASSE DE INTERESSE OUTLIER")
    print(f'Confusion matrix: \ntn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')

    # Calculating metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')

    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

print(f'\nCLASSE DE INTERESSE NORMAL \nConfusion matrix: \n{confusion_matrix_scorer(y_teste, y_pred_test)}')

print(f'Precisão: {precision:.2f}')
print(f'Revocação: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')