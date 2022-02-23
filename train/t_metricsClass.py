from typing import Any

class Metrics:
    def __init__(self):
        self.sessions = dict()
        self.metrics = set()
        self.current_session = None

    def add_session(self, staining, resize_ratio):
        key = f"{staining}_{resize_ratio}"
        if key not in self.sessions.keys():
            self.sessions[key] = dict()
        self.current_session = self.sessions[key]

    def print_session(self):
        for metric in self.current_session:
            print(f"{metric.upper()} = {self.current_session[metric]}")

    def add_metric(self, name: str, value: Any = None):
        if name not in self.current_session.keys():
            self.current_session[name] = []
            if name not in self.metrics:
                self.metrics.add(name)
        self.current_session[name].append(value)

    def mean(self, metric):
        vals = [i for s in self.sessions.keys() for i in self.sessions[s][metric]]
        return sum(vals) / len(vals)

    def averaged_metrics(self):
        average = dict()
        for metric in self.metrics:
            average[metric] = self.mean(metric)
        return average


if __name__ == '__main__':
    import random
    stainings = ["HE", "PAS", "PM"]
    resize_ratios = [3, 3]
    metrics = Metrics()
    for staining in stainings:
        for r_ratio in resize_ratios:
            metrics.add_session(staining, r_ratio)
            acc = random.uniform(0, 1)
            loss = random.uniform(0, 0.05)
            epochs = random.randint(1, 100)

            metrics.add_metric("accuracy", acc)
            metrics.add_metric("train_loss", loss)
            metrics.add_metric("num_epochs", epochs)

            metrics.print_session()
    print("\n\n")
    print(metrics.averaged_metrics())
