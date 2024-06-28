# model_evaluation.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    return acc, conf_mat, precision, recall, f1

if __name__ == "__main__":
    data = pd.read_csv('processed_data.csv')
    y = data['Label']
    X = data.drop('Label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    model = joblib.load('trained_model.joblib')
    acc, conf_mat, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{conf_mat}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
