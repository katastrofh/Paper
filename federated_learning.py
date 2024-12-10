
import numpy as np

class FederatedLearning:
    def __init__(self):
        self.global_model = np.random.rand(10)

    def aggregate_updates(self, updates):
        aggregated = np.mean(updates, axis=0)
        self.global_model = aggregated
        return self.global_model
    