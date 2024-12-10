
import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, data, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.data}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        target = '0' * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()
        print(f"Block mined: {self.hash} with nonce: {self.nonce}")


class Blockchain:
    def __init__(self):
        self.chain = []
        self.difficulty = 4  # Default difficulty level
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_block = Block(0, "0", {"message": "Genesis Block"})
        genesis_block.hash = hashlib.sha256("Genesis Block".encode('utf-8')).hexdigest()
        self.chain.append(genesis_block)

    def add_block(self, block):
        if self.is_valid_block(block):
            self.chain.append(block)
            return True
        return False

    def is_valid_block(self, block):
        valid_hash = block.previous_hash == self.chain[-1].hash
        valid_proof = block.hash.startswith('0' * self.difficulty)
        return valid_hash and valid_proof and self.is_valid_data(block.data)

    def is_valid_data(self, data):
        """Ensure the block data is correctly formatted."""
        return isinstance(data, dict) and 'updates' in data and 'global_model' in data
