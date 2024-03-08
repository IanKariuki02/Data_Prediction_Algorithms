import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
titanic_data = pd.read_csv('dataset/Titanic-Dataset.csv')
# for column in titanic_data.columns:
# unique_values = titanic_data[column].unique()
# print(f"Unique values in column '{column}': {unique_values}")

# Fill missing values in the 'Embarked' column with the most common value
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Extract titles from the 'Name' column
titanic_data['Title'] = titanic_data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

# Drop the 'Name' column
titanic_data.drop('Name', axis=1, inplace=True)
# Drop the 'Ticket' column
titanic_data.drop('Ticket', axis=1, inplace=True)
# Drop the 'Ticket' column
titanic_data.drop('Cabin', axis=1, inplace=True)

# Separate features and labels
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Handle categorical data
label_encoder = LabelEncoder()
X['Sex_encoded'] = label_encoder.fit_transform(X['Sex'])
X.drop('Sex', axis=1, inplace=True)  # Drop the original 'Sex' column

# One-hot encoding for 'Title' column
X = pd.concat([X, pd.get_dummies(X['Title'], prefix='Title')], axis=1)
X.drop('Title', axis=1, inplace=True)  # Drop the original 'Title' column

# One-hot encoding for 'Embarked' column
X = pd.get_dummies(X, columns=['Embarked'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = (y_test == y_pred).mean()
print("Accuracy:", accuracy)
