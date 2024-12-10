from flask import Flask, request, jsonify
import threading
import time
import requests

app = Flask(__name__)

peer_nodes = []  # List of peer nodes with their host and port


@app.route("/register_peer", methods=["POST"])
def register_peer():
    """Register a peer node with the registry."""
    try:
        data = request.get_json()
        if data["url"] not in peer_nodes:
            peer_nodes.append(data["url"])
            print(f"Peer registered: {data['url']}")
        return jsonify({"message": "Peer registered successfully"}), 200
    except Exception as e:
        print(f"Error in /register_peer: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/get_peers", methods=["GET"])
def get_peers():
    """Return the list of currently active peer nodes."""
    try:
        print("Serving /get_peers request")
        return jsonify({"peers": peer_nodes}), 200
    except Exception as e:
        print(f"Error in /get_peers: {e}")
        return jsonify({"error": str(e)}), 500


def monitor_peers():
    """Monitor the health of peer nodes and remove inactive ones."""
    while True:
        for peer in list(peer_nodes):  # Iterate over a copy to safely modify the list
            try:
                response = requests.get(f"{peer}/heartbeat", timeout=5)
                if response.status_code != 200:
                    raise Exception(f"Invalid response: {response.status_code}")
            except Exception as e:
                print(f"Peer {peer} is unreachable. Removing from the list: {e}")
                peer_nodes.remove(peer)
        time.sleep(10)  # Check every 10 seconds


if __name__ == "__main__":
    # Start the peer monitoring thread
    threading.Thread(target=monitor_peers, daemon=True).start()

    # Run the Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)
