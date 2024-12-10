import requests
import tensorflow as tf
import numpy as np
import time
import random
import argparse
from termcolor import colored  # For color-coded logs


class CarNode:
    def __init__(self, car_id, peer_nodes):
        self.car_id = car_id
        self.peer_nodes = peer_nodes
        self.local_model = self.initialize_model()
        self.local_data = self.generate_local_data()

    def generate_local_data(self):
        """Generate more realistic training data."""
        x_train = np.random.rand(1000, 10)  # More samples for better training
        y_train = (np.sum(x_train, axis=1) > 5).astype(int)  # A non-linear decision boundary
        print(colored(f"[{self.car_id}] Local data generated.", "cyan"))
        return x_train, y_train

    def initialize_model(self):
        """Use a deeper model architecture for better learning."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print(colored(f"[{self.car_id}] Model initialized successfully.", "green"))
        return model



    def train_model(self):
        x_train, y_train = self.local_data
        self.local_model.fit(x_train, y_train, epochs=1, verbose=0)
        print(colored(f"[{self.car_id}] Model trained for one epoch.", "yellow"))
        return self.local_model.get_weights()

    def submit_update(self):
        """Submit the update to a peer dynamically if accuracy improves."""
        current_accuracy = self.evaluate_global_model()
        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = current_accuracy

        if current_accuracy <= self.best_accuracy:
            print(colored(f"[{self.car_id}] Current model is not better than the previous best. Skipping update.", "yellow"))
            return

        self.best_accuracy = current_accuracy
        print(colored(f"[{self.car_id}] Submitting update as it improved the accuracy to {current_accuracy:.2f}.", "green"))

        weights = self.local_model.get_weights()
        serialized_weights = [w.tolist() for w in weights]
        model_update = {
            "car_id": self.car_id,
            "model_update": serialized_weights
        }

        if not self.peer_nodes:
            print(colored(f"[{self.car_id}] No peers available for update submission.", "red"))
            return

        designated_peer = self.peer_nodes[0]
        try:
            print(colored(f"[{self.car_id}] Trying to submit update to {designated_peer}...", "yellow"))
            response = requests.post(f"{designated_peer}/submit_update", json=model_update, timeout=5)
            response.raise_for_status()
            print(colored(f"[{self.car_id}] Update successfully submitted to {designated_peer}.", "green"))
        except Exception as e:
            print(colored(f"[{self.car_id}] Failed to submit update to {designated_peer}: {e}", "red"))



    def refresh_peer_nodes(self, registry_url):
        try:
            self.peer_nodes = get_peer_nodes(registry_url)
            print(colored(f"[{self.car_id}] Updated peer list: {self.peer_nodes}", "cyan"))
        except Exception as e:
            print(colored(f"[{self.car_id}] Failed to refresh peers: {e}", "red"))

    def fetch_global_model(self):
        """Fetch the global model from available peer nodes."""
        if not self.peer_nodes:
            print(colored(f"[{self.car_id}] No peers available to fetch global model.", "red"))
            return

        for peer in self.peer_nodes:
            try:
                print(colored(f"[{self.car_id}] Fetching global model from {peer}...", "yellow"))
                response = requests.get(f"{peer}/global_model", timeout=5)
                response.raise_for_status()
                serialized_weights = response.json().get("global_model")
                if serialized_weights:
                    self.local_model.set_weights([np.array(w) for w in serialized_weights])
                    print(colored(f"[{self.car_id}] Global model updated successfully from {peer}.", "green"))
                    return
            except requests.exceptions.RequestException as e:
                print(colored(f"[{self.car_id}] Failed to fetch global model from {peer}: {e}", "red"))

        print(colored(f"[{self.car_id}] Could not fetch global model from any peer.", "red"))





    def evaluate_global_model(self):
        """Evaluate the global model on local data."""
        x_test, y_test = self.local_data
        loss, accuracy = self.local_model.evaluate(x_test, y_test, verbose=0)
        print(colored(f"[{self.car_id}] Global model accuracy: {accuracy:.2f}", "blue"))
        return accuracy




def get_peer_nodes(registry_url):
    """Fetch peer nodes from the registry."""
    try:
        if not registry_url.endswith("/get_peers"):
            registry_url = f"{registry_url}/get_peers"
        response = requests.get(registry_url, timeout=5)
        response.raise_for_status()
        peers = response.json().get("peers", [])
        print(colored(f"Discovered peers: {peers}", "blue"))
        return peers
    except requests.exceptions.RequestException as e:
        print(colored(f"Error fetching peers: {e}", "red"))
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Node")
    parser.add_argument("--id", type=str, required=True, help="Car ID")
    parser.add_argument("--registry", type=str, required=True, help="Registry server URL")
    args = parser.parse_args()

    peers = []
    for _ in range(5):  # Retry up to 5 times
        peers = get_peer_nodes(args.registry)
        if peers:
            break
        print(colored(f"[{args.id}] Retrying to fetch peers...", "yellow"))
        time.sleep(5)

    if not peers:
        print(colored(f"[{args.id}] No peers found after retries. Exiting.", "red"))
        exit(1)

    car_node = CarNode(args.id, peers)

    while True:
        car_node.refresh_peer_nodes(args.registry)  # Refresh the list of peer nodes
        car_node.train_model()                      # Train the local model
        car_node.submit_update()                    # Submit local updates to a peer node

        car_node.fetch_global_model()               # Fetch the latest global model from peers
        car_node.evaluate_global_model()            # Evaluate the global model on local data

        time.sleep(10)  # Sleep before the next iteration
