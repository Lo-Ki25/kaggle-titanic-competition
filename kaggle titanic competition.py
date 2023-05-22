import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the training data
train_data = pd.read_csv('train.csv')

# Load the test data
test_data = pd.read_csv('test.csv')

# Data preprocessing
# Drop unnecessary columns
train_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)

# Convert categorical variables to numerical
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Align test data columns with training data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Split the training data into features and target
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imputed, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_imputed)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Exclude the target variable from the test data
test_data_without_target = test_data.drop('Transported', axis=1)

# Impute missing values in the test data
test_data_imputed = imputer.transform(test_data_without_target)

# Make predictions on the test set
test_predictions = model.predict(test_data_imputed)

# Prepare the submission file
submission = pd.DataFrame({'PassengerId': range(1, len(test_data) + 1), 'Transported': test_predictions})

# Convert 'PassengerId' column to string
submission['PassengerId'] = submission['PassengerId'].astype(str)

# Save the submission file
submission.to_csv('submission.csv', index=False)

# Display the results
print("Submission file generated successfully.")
print("Number of test predictions:", len(test_predictions))
print("Sample predictions:")
print(submission.head())
