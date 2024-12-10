from flask import Flask, request, jsonify
from blockchain import Blockchain, Block
import threading
import requests
import time
import argparse
import socket
import hashlib
from termcolor import colored
import numpy as np
import tensorflow as tf

app = Flask(__name__)


def get_host_ip():
    """Get the actual IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(colored(f"Failed to determine host IP: {e}", "red"))
        return "127.0.0.1"


def register_with_registry(node_url, registry_url):
    """Register the peer node with the central registry."""
    try:
        if not registry_url.endswith("/register_peer"):
            registry_url = f"{registry_url}/register_peer"
        response = requests.post(registry_url, json={"url": node_url}, timeout=5)
        response.raise_for_status()
        print(colored(f"Registered with registry at {registry_url}.", "green"))
    except requests.exceptions.RequestException as e:
        print(colored(f"Failed to register with registry: {e}", "red"))


def fetch_peers_from_registry(registry_url):
    """Fetch the list of peer nodes from the registry server."""
    try:
        response = requests.get(f"{registry_url}/get_peers", timeout=5)
        response.raise_for_status()
        peers = response.json().get("peers", [])
        print(colored(f"Peers fetched: {peers}", "cyan"))
        return peers
    except requests.exceptions.RequestException as e:
        print(colored(f"Error fetching peers: {e}", "red"))
        return []


class PeerNode:
    def __init__(self, node_id, host, port, peer_nodes, registry_url):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peer_nodes = peer_nodes  # This is initially fetched from the registry
        self.registry_url = registry_url
        self.blockchain = Blockchain()
        self.local_updates = []
        self.received_transactions = set()  # Prevent rebroadcasting transactions
        self.received_blocks = set()  # Prevent rebroadcasting blocks
        self.global_model = None

        # Start the heartbeat mechanism in a separate thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.heartbeat_thread.start()

        # Start the periodic synchronization thread
        self.synchronization_thread = threading.Thread(target=self.periodic_synchronization, daemon=True)
        self.synchronization_thread.start()

        # Start the periodic consensus thread
        self.consensus_thread = threading.Thread(target=self.periodic_consensus_check, daemon=True)
        self.consensus_thread.start()

    def periodic_consensus_check(self):
        """Periodically check if consensus conditions are met."""
        while True:
            if len(self.local_updates) >= 5:  # Adjust threshold as needed
                print(colored(f"Peer {self.node_id}: Enough updates collected, initiating consensus.", "yellow"))
                self.aggregate_updates()
                self.perform_consensus()
            time.sleep(10)  # Check every 10 seconds
    def validate_update(self, update):
        """Validate the structure and content of a local update."""
        try:
            if not isinstance(update, dict) or "car_id" not in update or "model_update" not in update:
                raise ValueError("Invalid update structure.")
            print(colored(f"Peer {self.node_id}: Update from Car {update['car_id']} validated.", "green"))
            return True
        except ValueError as e:
            print(colored(f"Peer {self.node_id}: Update validation failed - {e}", "red"))
            return False

    def heartbeat(self):
        """Check the health of peer nodes and fetch updates from the registry."""
        while True:
            # Remove unreachable peers from the internal list
            for peer in list(self.peer_nodes):  # Iterate over a copy of the peer list
                try:
                    response = requests.get(f"{peer}/heartbeat", timeout=5)
                    if response.status_code != 200:
                        raise Exception("Invalid response")
                except Exception as e:
                    print(colored(f"Peer {self.node_id}: Peer {peer} is unreachable. Removing it.", "red"))
                    self.peer_nodes.remove(peer)

            # Fetch the latest peers from the registry
            try:
                updated_peers = fetch_peers_from_registry(self.registry_url)
                # Replace the internal peer list with the updated one from the registry
                self.peer_nodes = [peer for peer in updated_peers if peer != f"http://{self.host}:{self.port}"]
                print(colored(f"Peer {self.node_id}: Updated peers: {self.peer_nodes}", "blue"))
            except Exception as e:
                print(colored(f"Peer {self.node_id}: Failed to fetch peers from registry: {e}", "red"))

            time.sleep(10)  # Check every 10 seconds

    def add_local_update(self, update):
        """Handle a new local update."""
        update_id = hashlib.sha256(str(update).encode()).hexdigest()

        if update_id in self.received_transactions:
            print(colored(f"Peer {self.node_id}: Duplicate transaction skipped.", "yellow"))
            return
        if not self.validate_update(update):
            return

        # Debug incoming update structure
        print(colored(f"Peer {self.node_id}: Update structure: {update}", "cyan"))

        self.local_updates.append(update)
        self.received_transactions.add(update_id)
        print(colored(f"Peer {self.node_id}: Update added to local list.", "blue"))

        self.broadcast_transaction(update)

        if len(self.local_updates) >= 5:
            self.aggregate_updates()
            self.perform_consensus()


    def aggregate_updates(self):
        """Aggregate updates using Federated Averaging."""
        try:
            weights = [update["model_update"] for update in self.local_updates]

            # Validate structure of weights
            if not all(isinstance(w, list) and all(isinstance(layer, list) for layer in w) for w in weights):
                raise ValueError("Model updates have inconsistent or invalid structure.")

            # Aggregate weights layer by layer
            aggregated_weights = [
                [sum(neuron_weights) / len(neuron_weights) for neuron_weights in zip(*layers)]
                for layers in zip(*weights)
            ]

            self.global_model = aggregated_weights
            print(colored(f"Peer {self.node_id}: Global model aggregated successfully.", "green"))
        except Exception as e:
            print(colored(f"Aggregation error in Peer {self.node_id}: {e}", "red"))
            self.global_model = None





    def perform_consensus(self):
        """Perform consensus and broadcast the new block."""
        if len(self.local_updates) < 5:
            print(colored(f"Peer {self.node_id}: Not enough updates for consensus. Skipping.", "red"))
            return

        if not self.global_model:
            print(colored(f"Peer {self.node_id}: No global model available. Skipping consensus.", "red"))
            return

        retry_attempts = 3
        for attempt in range(retry_attempts):
            if self.validate_aggregated_model():
                break
            print(colored(f"Peer {self.node_id}: Retrying model validation ({attempt + 1}/{retry_attempts})...", "yellow"))
        else:
            print(colored(f"Peer {self.node_id}: Aggregated model failed validation. Aborting consensus.", "red"))
            return

        new_block = Block(
            index=len(self.blockchain.chain),
            previous_hash=self.blockchain.chain[-1].hash,
            data={"updates": self.local_updates, "global_model": self.global_model},
        )
        new_block.mine_block(self.blockchain.difficulty)

        if self.blockchain.add_block(new_block):
            print(colored(f"Peer {self.node_id}: Block added to blockchain: {new_block.hash}", "green"))
            self.broadcast_block(new_block)
            print(colored(f"Peer {self.node_id}: Consensus performed successfully.", "green"))
        else:
            print(colored(f"Peer {self.node_id}: Block rejected during consensus.", "red"))

        self.local_updates = []  # Clear updates after successful consensus







    def broadcast_transaction(self, transaction):
        """Broadcast a transaction to all peer nodes."""
        for peer in self.peer_nodes:
            try:
                requests.post(f"{peer}/receive_transaction", json=transaction, timeout=5)
            except Exception:
                continue

    def broadcast_block(self, block):
        """Broadcast a block to all peer nodes."""
        for peer in self.peer_nodes:
            try:
                requests.post(f"{peer}/receive_block", json={"block": vars(block)}, timeout=5)
            except Exception:
                continue
    def validate_aggregated_model(self):
        """Validate the aggregated model before broadcasting."""
        if not self.global_model:
            print(colored(f"Peer {self.node_id}: No global model to validate.", "red"))
            return False

        # Use synthetic test data
        x_test = np.random.rand(100, 10)
        y_test = (np.sum(x_test, axis=1) > 5).astype(int)

        # Load the aggregated weights into a temporary model
        temp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        temp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        temp_model.set_weights([np.array(w) for w in self.global_model])

        loss, accuracy = temp_model.evaluate(x_test, y_test, verbose=0)
        print(colored(f"Peer {self.node_id}: Aggregated model accuracy: {accuracy:.2f}", "green"))
        return accuracy > 0.75  # Adjust threshold if necessary

    def synchronize_updates(self):
        """Fetch and reconcile updates from other peers."""
        for peer in self.peer_nodes:
            try:
                response = requests.get(f"{peer}/get_updates", timeout=5)
                response.raise_for_status()
                peer_updates = response.json().get("local_updates", [])

                for update in peer_updates:
                    # Avoid duplicate updates
                    update_id = hashlib.sha256(str(update).encode()).hexdigest()
                    if update_id not in self.received_transactions:
                        if self.validate_update(update):
                            self.local_updates.append(update)
                            self.received_transactions.add(update_id)
                            print(colored(f"Peer {self.node_id}: Synchronized update from {peer}.", "green"))
            except Exception as e:
                print(colored(f"Peer {self.node_id}: Failed to synchronize updates from {peer}: {e}", "red"))
    def periodic_synchronization(self):
        """Periodically synchronize updates with other peers."""
        while True:
            self.synchronize_updates()
            time.sleep(15)  # Synchronize every 15 seconds



peer_node = None


@app.route("/submit_update", methods=["POST"])
def submit_update():
    data = request.get_json()
    peer_node.add_local_update(data)
    return jsonify({"message": "Update received"}), 200


@app.route("/receive_transaction", methods=["POST"])
def receive_transaction():
    data = request.get_json()
    peer_node.add_local_update(data)
    return jsonify({"message": "Transaction received"}), 200


@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    return jsonify({"status": "alive"}), 200


@app.route("/receive_block", methods=["POST"])
def receive_block():
    data = request.get_json()
    peer_node.receive_block(data["block"])
    return jsonify({"message": "Block received"}), 200

@app.route("/global_model", methods=["GET"])
def get_global_model():
    """Provide the current global model to car nodes."""
    if peer_node.global_model is None:
        print(colored("Global model not available to serve.", "red"))
        return jsonify({"error": "Global model not available"}), 503  # Use 503 for service unavailable

    print(colored("Global model served successfully.", "green"))
    return jsonify({"global_model": peer_node.global_model}), 200

@app.route("/get_updates", methods=["GET"])
def get_updates():
    """Provide the current local updates to other peers."""
    return jsonify({"local_updates": peer_node.local_updates}), 200





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peer Node")
    parser.add_argument("--id", type=str, required=True, help="Node ID")
    parser.add_argument("--port", type=int, required=True, help="Port")
    parser.add_argument("--registry", type=str, required=True, help="Registry URL")
    args = parser.parse_args()

    host_ip = get_host_ip()
    node_url = f"http://{host_ip}:{args.port}"
    register_with_registry(node_url, args.registry)

    peers = fetch_peers_from_registry(args.registry)
    peer_node = PeerNode(args.id, host_ip, args.port, peers, args.registry)
    app.run(host="0.0.0.0", port=args.port, debug=False)
