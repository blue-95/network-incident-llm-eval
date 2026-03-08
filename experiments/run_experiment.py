import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.preprocess import load_data
from xgboost import XGBClassifier
from evaluation.evaluator import evaluate_model


X_train, X_test, y_train, y_test, le = load_data()

model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)

results = evaluate_model("xgboost", y_test, preds)

print(results)