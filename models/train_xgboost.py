import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xgboost import XGBClassifier
from pipeline.preprocess import load_data
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test, le = load_data()

model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))