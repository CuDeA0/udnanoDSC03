"""main.py
Wrap the execution into a python function
"""
import os

if __name__ == "__main__":
    os.system("python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db")
    os.system("python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl")
