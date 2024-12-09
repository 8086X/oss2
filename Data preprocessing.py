# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the Titanic dataset (from Kaggle sample dataset URL or placeholder here)
# Placeholder data - create a simple Titanic-like dataset
data = {
    'Pclass': [1, 3, 2, 3, 1, 3, 2, 3, 1, 2],
    'Sex': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'female'],
    'Age': [22, 38, 26, 35, 28, 2, 40, 27, 19, 36],
    'SibSp': [1, 1, 0, 0, 0, 4, 0, 0, 0, 1],
    'Parch': [0, 0, 0, 0, 0, 1, 0, 0, 0, 2],
    'Fare': [7.25, 71.2833, 7.925, 8.05, 53.1, 21.075, 13.0, 11.1333, 30.0, 23.45],
    'Survived': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Data preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Encode 'Sex' column
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Displaying accuracy
accuracy
