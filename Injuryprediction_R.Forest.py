import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/injury_data.csv")

print(data.info())
print(data.describe())

features = ["Player_Age", "Player_Height", "Player_Weight", "Previous_Injuries", "Training_Intensity"]
target = "Likelihood_of_Injury"

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

predictions_all = model.predict(data[features])
predictions_all_str = np.where(predictions_all == 0, "No", "Yes")
predictions_series = pd.Series(predictions_all_str, index=data.index)
age_with_predictions = pd.concat([data['Player_Age'], predictions_series], axis=1)

age_with_predictions.columns = ['Player_Age', 'Predicted_Likelihood_of_Injury']

print(age_with_predictions.head(10))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

data[['Player_Age', 'Player_Height', 'Player_Weight', 'Previous_Injuries', 'Likelihood_of_Injury']].plot(kind='hist', bins=10)
plt.xlabel('Player ages')
plt.ylabel('Frequency')
plt.title('Distribution of Players')
plt.show()
