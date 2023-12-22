# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sys
# import scipy
# from sklearn.metrics import f1_score


# # load the dataset using pandas
# data = pd.read_csv('creditcard.csv')

# # dataset exploring
# print(data.columns)

# # Print the shape of the data
# data = data.sample(frac=0.1, random_state = 1)
# print(data.shape)
# print(data.describe())

# # V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

# # Plot histograms of each parameter 
# data.hist(figsize = (20, 20))
# plt.show()

# # Determine number of fraud cases in dataset

# Fraud = data[data['Class'] == 1]
# Valid = data[data['Class'] == 0]

# outlier_fraction = len(Fraud)/float(len(Valid))
# print(outlier_fraction)

# print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
# print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# # Correlation matrix
# corrmat = data.corr()
# fig = plt.figure(figsize = (12, 9))

# sns.heatmap(corrmat, vmax = .8, square = True)
# plt.show()

# # Get all the columns from the dataFrame
# columns = data.columns.tolist()

# # Filter the columns to remove data we do not want
# columns = [c for c in columns if c not in ["Class"]]

# # Store the variable we'll be predicting on
# target = "Class"

# X = data[columns]
# Y = data[target]

# # Print shapes
# print(X.shape)
# print(Y.shape)

# from sklearn.metrics import classification_report, accuracy_score

# from sklearn.linear_model import LogisticRegression
# # Assuming 'X' is your feature set and 'Y' is the label set

# # Import the necessary libraries
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import StandardScaler

# # Standardize the features (very important for logistic regression)
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X)

# # Initialize the Logistic Regression model
# # Note: The option class_weight='balanced' can help with the imbalance of the classes
# logistic_model = LogisticRegression(class_weight='balanced')

# # Fit the model
# logistic_model.fit(X_standardized, Y)

# # Prediction
# Y_pred = logistic_model.predict(X_standardized)

# # Error
# n_errors = (Y_pred != Y).sum()

# # Evaluation metrics
# print(f"Number of errors: {n_errors}")
# print(f"Accuracy Score: {accuracy_score(Y, Y_pred)}")
# print(classification_report(Y, Y_pred))
# print(accuracy_score(Y, Y_pred))
# print(classification_report(Y, Y_pred))
# print('F1 Score:', f1_score(Y, Y_pred))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

# load the dataset using pandas
data = pd.read_csv('creditcard.csv')

# Sample the dataset
data = data.sample(frac=0.1, random_state=1)

# Print the shape of the data
print(data.shape)

# Print basic statistics
print(data.describe())

# Plot histograms of each parameter (optional, can be commented out for quicker execution)
data.hist(figsize=(20, 20))
plt.show()

# Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

# Calculate the outlier fraction
outlier_fraction = len(Fraud) / float(len(Valid))
print('Outlier fraction:', outlier_fraction)

# Print the fraud and valid cases
print('Fraud Cases:', len(Fraud))
print('Valid Transactions:', len(Valid))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Prepare the data for training/testing
columns = [c for c in data.columns if c not in ['Class']]
X = data[columns]
Y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Apply SMOTE only on training data to address class imbalance
smote = SMOTE(random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Logistic Regression
logistic_model = LogisticRegression(class_weight='balanced', solver='liblinear')
logistic_model.fit(X_train_resampled, Y_train_resampled)

# Prediction probabilities
Y_probs = logistic_model.predict_proba(X_test)[:, 1]

# Adjust the decision threshold
precision, recall, thresholds = precision_recall_curve(Y_test, Y_probs)
threshold = thresholds[np.argmax(2*recall[:-1]*precision[:-1]/(recall[:-1]+precision[:-1]))]

Y_pred = (Y_probs >= threshold).astype(int)

# Printing the classification report
print(classification_report(Y_test, Y_pred))

# Printing the F1 score
print('F1 Score:', f1_score(Y_test, Y_pred))

# Optionally print the accuracy score
print('Accuracy Score:', accuracy_score(Y_test, Y_pred))
#done