LLM Evaluation Framework for Network Incident Diagnosis

This project implements a production-style machine learning pipeline for diagnosing simulated network incidents using telemetry data. The system compares traditional machine learning models and transformer-based architectures and evaluates their performance using a modular evaluation framework.

The goal of the project is to simulate how AI systems can analyze network telemetry and identify root causes of performance issues such as DNS failures, routing changes, and packet loss.

⸻

System Architecture

Telemetry Data Generation
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
   ├── XGBoost (baseline ML model)
   └── DistilBERT (transformer classifier)
        ↓
Evaluation Framework
        ↓
Experiment Tracking


Key Features
	•	Synthetic telemetry dataset generation for network incidents
	•	Feature engineering for latency, packet loss, jitter, DNS errors, and routing changes
	•	Baseline ML model using XGBoost
	•	Transformer-based classifier using DistilBERT
	•	Modular evaluation framework computing:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 score
	•	Confusion matrix
	•	Structured experiment logging for reproducibility

⸻

Technologies Used
	•	Python
	•	PyTorch
	•	Transformers
	•	XGBoost
	•	scikit-learn
	•	pandas / numpy

⸻

Results

Two models were evaluated on the telemetry classification task.


Model         Accuracy  Precision  Recall  F1 Score
XGBoost        0.995     0.995      0.995   0.994
DistilBERT     0.915     0.859      0.915   0.881


Observations:
	•	Tree-based models perform extremely well on structured telemetry data.
	•	Transformer models remain competitive when structured features are converted to text representations.
	•	This comparison demonstrates the importance of selecting model architectures aligned with the underlying data modality.

Future Improvements
	•	Evaluate larger transformer architectures
	•	Integrate real network telemetry datasets
	•	Add model monitoring and experiment dashboards
	•	Implement automated ML pipelines

