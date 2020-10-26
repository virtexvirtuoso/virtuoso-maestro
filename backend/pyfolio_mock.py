# Mock pyfolio module to bypass installation issues
class MockPyfolio:
    def __init__(self):
        pass
    
    def create_full_tear_sheet(self, *args, **kwargs):
        return {}
    
    def create_returns_tear_sheet(self, *args, **kwargs):
        return {}
    
    def create_position_tear_sheet(self, *args, **kwargs):
        return {}
    
    def create_txn_tear_sheet(self, *args, **kwargs):
        return {}

# Create mock functions
def create_full_tear_sheet(*args, **kwargs):
    return {}

def create_returns_tear_sheet(*args, **kwargs):
    return {}

def create_position_tear_sheet(*args, **kwargs):
    return {}

def create_txn_tear_sheet(*args, **kwargs):
    return {} 