# model_training.py
# model_evaluation.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split  # Ensure this line is present
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
# Rest of your code...


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    forest = RandomForestClassifier(max_depth=5)
    forest.fit(X_train, y_train)
    return forest

if __name__ == "__main__":
    data = pd.read_csv('processed_data.csv')
    y = data['Label']
    X = data.drop('Label', axis=1)
    model = train_model(X, y)
    joblib.dump(model, 'trained_model.joblib')  # Save trained model for next step
