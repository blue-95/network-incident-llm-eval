import pandas as pd
import numpy as np

np.random.seed(42)

N = 1000

data = {
    "latency": np.random.normal(120, 50, N),
    "packet_loss": np.random.uniform(0, 0.1, N),
    "jitter": np.random.normal(20, 10, N),
    "dns_error": np.random.randint(0, 2, N),
    "route_change": np.random.randint(0, 2, N)
}

df = pd.DataFrame(data)

def assign_label(row):

    if row["dns_error"] == 1:
        return "dns_failure"

    if row["route_change"] == 1:
        return "routing_issue"

    if row["packet_loss"] > 0.07:
        return "packet_drop"

    if row["latency"] > 200:
        return "congestion"

    return "normal"

df["label"] = df.apply(assign_label, axis=1)

df.to_csv("data/telemetry_logs.csv", index=False)

print("Dataset generated.")