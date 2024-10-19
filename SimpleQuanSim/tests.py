import unittest
from simple_sim import *


class test_indexes(unittest.TestCase):
    
    def test_get_by_index(self):
        b = '11010101'
        # positions = list(range(len(b)))
        for pos in range(len(b)):
            self.assertEqual(get_by_index(int(b, 2), pos), int(b[-pos-1])) 

    def test_get_by_indexes(self):
        b = '11010101'
        positions = list(range(len(b)))
        # for pos in range(len(b)):
        self.assertEqual(get_by_indexes(int(b, 2), positions), int(b,2)) 
        
    def test_upgrade_index(self):
        b_start = '1111010'
        bit = 0
        pos = 4
        b_end = '1101010'
        self.assertEqual(upgrade_index(int(b_start, 2), pos, bit), int(b_end, 2)) 
        self.assertEqual(upgrade_index(int(b_end, 2), pos, bit), int(b_end, 2)) 
        
        b_start = '1101010'
        bit = 1
        pos = 4
        b_end = '1111010'
        self.assertEqual(upgrade_index(int(b_start, 2), pos, bit), int(b_end, 2)) 
        self.assertEqual(upgrade_index(int(b_end, 2), pos, bit), int(b_end, 2)) 
        
    def test_upgrade_indexes(self):
        b_start = '1111010'
        val = 2
        positions = [1,2,4]
        b_end = '1101100'
        self.assertEqual(upgrade_indexes(int(b_start, 2), positions, val), int(b_end, 2)) 
        self.assertEqual(upgrade_indexes(int(b_end, 2), positions, val), int(b_end, 2)) 
    
class test_state_evolution(unittest.TestCase):
    def test_one_qubit(self):
        s = State(1,[1,0])
        op = Operator([0], [1,1,1,-1])
        s_evolve = s.get_evolved_state(op)
        s_end = State(1, [1,1])
        self.assertFalse(not all([i==j for i, j in zip(s_evolve, s_end)]), msg=str(s_evolve.get_evolved_state(op).data))
    
        s = State(1,[0,1])
        op = Operator([0], [1,1,1,-1])
        s_evolve = s.get_evolved_state(op)
        s_end = State(1, [1,-1])
        self.assertFalse(not all([i==j for i, j in zip(s_evolve, s_end)]), msg=str(s_evolve.get_evolved_state(op).data))
       
        s = State(1,[1,0])
        op = Operator([0], [0,1,1,0])
        s_evolve = s.get_evolved_state(op)
        s_end = State(1, [0,1])
        print(s_evolve.data)
        self.assertFalse(not all([i==j for i, j in zip(s_evolve, s_end)]), msg=str(s_evolve.data))
        self.assertFalse(not all([i==j for i, j in zip(s_evolve.get_evolved_state(op), s)]), msg=str(s_evolve.get_evolved_state(op).data))
    
    def test_many_qubit(self):
        s = State(3,[1,0,0,0,0,0,0,0]) # |000>
        op0 = Operator([0], [1,1,1,-1])
        op1 = Operator([1], [1,1,1,-1])
        op2 = Operator([2], [1,1,1,-1])
        
        s_end = State(3, [1,0,0,0,1,0,0,0]) # |000> + |100>
        self.assertFalse(not all([i==j for i, j in zip(s.get_evolved_state(op0), s_end)]), msg=str(s.get_evolved_state(op0).data))
        
        s_end = State(3, [1,0,1,0,0,0,0,0])# |000> + |010>
        self.assertFalse(not all([i==j for i, j in zip(s.get_evolved_state(op1), s_end)]), msg=str(s.get_evolved_state(op1).data))
        
        s_end = State(3, [1,1,0,0,0,0,0,0])# |000> + |001>
        self.assertFalse(not all([i==j for i, j in zip(s.get_evolved_state(op2), s_end)]), msg=str(s.get_evolved_state(op2).data))
    
    def test_cx_qubit(self):
        cx = Operator([1,2], [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0])
        s00 = State(3,[1,0,0,0,0,0,0,0]) # |000>
        s01 = State(3,[0,1,0,0,0,0,0,0]) # |001>
        s10 = State(3,[0,0,1,0,0,0,0,0]) # |010>
        s11 = State(3,[0,0,0,1,0,0,0,0]) # |011>
        self.assertFalse(not all([i==j for i, j in zip(s00.get_evolved_state(cx), s00)]), msg=str(s00.get_evolved_state(cx).data))
        self.assertFalse(not all([i==j for i, j in zip(s01.get_evolved_state(cx), s01)]), msg=str(s01.get_evolved_state(cx).data))
        self.assertFalse(not all([i==j for i, j in zip(s10.get_evolved_state(cx), s11)]), msg=str(s10.get_evolved_state(cx).data))
        self.assertFalse(not all([i==j for i, j in zip(s11.get_evolved_state(cx), s10)]), msg=str(s11.get_evolved_state(cx).data))

    def test_random(self):
        # s = State(3, )
        pass 
        
if __name__ == "__main__":
    unittest.main()