# data_loading.py
import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(['Domain'], axis=1).copy()
    return data

if __name__ == "__main__":
    filename = '5.urldata.csv'
    data = load_data(filename)
    data.to_csv('processed_data.csv', index=False)  # Save processed data for next step
