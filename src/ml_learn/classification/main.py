import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class PasteurizerModel:
  def __init__(self, data_path):
    self.data_path = data_path
    self.df = None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    self.dt_model = None
    self.rf_model = None

  def load_and_preprocess_data(self):
    # Load data
    self.df = pd.read_csv(self.data_path)
    
    # Remove missing values
    self.df = self.df.dropna()
    
    # Remove outliers
    state_column = 'MIXA_PASTEUR_STATE'
    self.df = self.df[self.df[state_column] < 2]
    
    # Split data into features and target
    X = self.df.iloc[:, 1:5].values
    y = self.df.iloc[:, -1:].values
    y = np.where(y == 'OK', 1, 0).ravel()
    
    # Split data into training and testing sets
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1)

  def train_decision_tree(self):
    self.dt_model = DecisionTreeClassifier(max_depth=3)
    self.dt_model.fit(self.X_train, self.y_train)

  def train_random_forest(self):
    self.rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
    self.rf_model.fit(self.X_train, self.y_train)

  def evaluate_model(self, model):
    y_pred = model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    report = classification_report(self.y_test, y_pred)
    return accuracy, report

  def visualize_decision_tree(self):
    plt.figure(figsize=(12, 8))
    plot_tree(self.dt_model, filled=True, feature_names=self.df.columns[1:5])
    plt.show()

  def run(self):
    self.load_and_preprocess_data()
    self.train_decision_tree()
    self.train_random_forest()
    
    dt_accuracy, dt_report = self.evaluate_model(self.dt_model)
    rf_accuracy, rf_report = self.evaluate_model(self.rf_model)
    
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print("Decision Tree Classification Report:")
    print(dt_report)
    
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print("Random Forest Classification Report:")
    print(rf_report)
    
    self.visualize_decision_tree()

def run_decision_tree():
  data_path = ""
  model = PasteurizerModel(data_path)
  model.run()

def run_ramdom_forest():
  data_path = ""
  model = PasteurizerModel(data_path)
  model.run()
