
from blockchain import Blockchain, Block

class ValidatorNode:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def validate_update(self, update):
        # Simplified validation logic
        return len(update['model_update']) == 10

    def create_block(self, update):
        previous_hash = self.blockchain.chain[-1].hash
        new_block = Block(len(self.blockchain.chain), previous_hash, update)
        new_block.mine_block(self.blockchain.difficulty)
        return new_block

    def validate_and_add_block(self, update):
        if self.validate_update(update):
            new_block = self.create_block(update)
            return self.blockchain.add_block(new_block)
        return False
    