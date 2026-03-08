import json
import time

from evaluation.metrics import compute_metrics


def evaluate_model(model_name, y_true, y_pred):

    start = time.time()

    metrics = compute_metrics(y_true, y_pred)

    latency = time.time() - start

    results = {
        "model": model_name,
        "metrics": metrics,
        "evaluation_latency": latency
    }

    with open("results/experiment_results.json", "a") as f:
        json.dump(results, f)
        f.write("\n")

    return results