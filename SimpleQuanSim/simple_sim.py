class Tensor:
    def __init__(self, n: list[int], array: list = None):
        self.n = n
        if array is None:
            self._data = [0]*(1<<n)
        else:
            self.data = array

    @property
    def data(self):
        return self._data
         
    @data.setter
    def data(self, array):
        if len(array) == 1<<self.n:
            self._data = array
        else:
            raise BufferError(f"Length {1<<self.n} is not equal to {len(array)}")
    
    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, val):
        self.data[index] = val
        
    def __repr__(self):
        s = str(self.n) + ": \n"
        return s + str(self.data) + "\n"
    
def get_by_index(index: int, pos: int):
    return (index >> pos) & 1

def get_by_indexes(index: int, positions: list[int]):
    x = 0
    for pos in reversed(positions):
        x = (x<<1) + get_by_index(index,pos)
    return x

def upgrade_index(index: int, pos, bit):
    if bit:
        return index | (1<<pos)
    else:
        return index & ~(1<<pos)

def upgrade_indexes(index: int, positions: list[int], val):
    for i, pos in enumerate(positions):
        index = upgrade_index(index, pos, get_by_index(val, i))
    return index


class Operator(Tensor):
    def __init__(self, qubits, array):
        self.qubits = qubits
        super().__init__(len(qubits)*2, array)
        
    def get_item(self, new, old, kron=True):
        if kron:
            return self[(new << (self.n//2)) +  old]
        else:
            # index = 0
            # for i in range(self.n//2 - 1, -1, -1):
            #     index = (index << 1) + get_by_index(new,i)
            #     index = (index << 1) + get_by_index(old,i)
            # return self[index]
            pass
     
class State(Tensor):
    def get_evolved_state(self, op: Operator):
        new_state = State(self.n)
        reversed_qubits = [self.n - 1 - i for i in reversed(op.qubits)]
        for index_state in range(1 << self.n):
            val = 0
            for index_op in range(1 << op.n//2):
                val += op.get_item(get_by_indexes(index_state, reversed_qubits),  index_op)\
                                *self[upgrade_indexes(index_state, reversed_qubits, index_op)]
            new_state[index_state] = val
        return new_state
    
    
if __name__ == "__main__":
    pass
    