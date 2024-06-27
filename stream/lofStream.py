import numpy as np
from sklearn.model_selection import train_test_split
from pyod.models.lof import LocalOutlierFactor
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import ArrayStreamer
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# Load the specific CICIDS2017 dataset
data, meta = arff.loadarff('data/Friday.arff')
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
print(f"Length x_normais: {len(X_normais)}")
print(f"Length x_outliers: {len(X_outliers)}")

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_normais, y_normais, test_size=0.3, random_state=42, shuffle=True)
print(f"Vetor treino: {len(X_train)}")
print(f"Vetor teste: {len(X_test)}")

# dfY_test = pd.DataFrame(y_test)
# dfY_test.to_csv('y_test_stream.csv', index=False)

dfTrain = pd.DataFrame(X_train)

X_train = dfTrain.to_numpy()
# dfTest = pd.DataFrame(X_test)
# dfTest.to_csv('X_test.csv', index=False)
# dfTrain.to_csv('X_train.csv', index=False)

#-------------------------------------------

y_teste = pd.DataFrame(y_test)
y_teste = y_teste.values

# Concatenate the outliers in the test vector   
X_test = np.concatenate((X_test, X_outliers), axis=0)
print(f"Vector test after concat: {len(X_test)}")
y_teste = np.append(y_teste, np.ones(len(X_outliers)))


"""## Train Test"""

# Initialization of the models
model = ReferenceWindowModel(model_cls=LocalOutlierFactor, window_size=500, sliding_size=200, initial_window_X=X_train[:100], novelty=True)
preprocessor = InstanceUnitNormScaler()  # Normalizer
postprocessor = RunningAveragePostprocessor(window_size=100)  # Running average postprocessor
calibrator = GaussianTailProbabilityCalibrator(running_statistics=True, window_size=1000)  # Probability calibrator

# Detect anomalies on streaming data
y_pred = []
iterator = ArrayStreamer(shuffle=False)  # Streamer to simulate streaming data

for i, (X, y) in tqdm(enumerate(iterator.iter(X_test, y_teste))):
    X = preprocessor.fit_transform_partial(X)
    anomaly_score = model.fit_score_partial(X)
    # anomaly_score = postprocessor.fit_transform_partial(anomaly_score)
    # calibrated_score = calibrator.fit_transform_partial(anomaly_score)
    y_pred.append(anomaly_score)

dfY_pred = pd.DataFrame(y_pred)
dfY_pred.to_csv('y_pred.csv', index=False)

# Define a threshold (valor que é escolhido para determinar anomalia)
threshold = 0.5

# Classify samples based on the threshold
y_pred = [1 if anomaly_score >= threshold else 0 for anomaly_score in y_pred]
dfY_pred = pd.DataFrame(y_pred)
dfY_pred.to_csv('y_predPosClass.csv', index=False)

"""## Plot"""

# Plot the results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y_pred[0:], label='prediction', linewidth=1)
ax.plot(y_teste[0:], label='ground truth', linewidth=3, linestyle='--')
ax.set_xlabel('time')
ax.set_ylabel('anomaly likelihood')
# ax.set_title(f'CICIDS2017 Dataset with {1e2*np.count_nonzero(y_all) / len(y_all):.2f}% outliers')
ax.legend()

# Save the plot to a file
plt.savefig("cicids.svg")

y_teste = pd.DataFrame(y_teste)
y_pred = pd.DataFrame(y_pred)

# Function lambda to transform 0 to 1 and 1 to -1
transform_function = lambda x: 1 if x == 0 else -1

# Apply the lambda function to the entire DataFrame
y_teste = y_teste.applymap(transform_function)
y_pred = y_pred.applymap(transform_function)

# y_teste are the actual labels, y_pred_teste are the model predictions
precision = precision_score(y_teste, y_pred)
recall = recall_score(y_teste, y_pred)
f1 = f1_score(y_teste, y_pred)

def confusion_matrix_scorer(y_teste, y_pred):
    cm = confusion_matrix(y_teste, y_pred)

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
    # accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')

    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

print(f'\nCLASSE DE INTERESSE NORMAL \nConfusion matrix: \n{confusion_matrix_scorer(y_teste, y_pred)}')

print(f'Precisão: {precision:.2f}')
print(f'Revocação: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')