import unittest
import torch 
import torchutils.low_rank as lr 

class LowRankTestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b, atol=1e-4))
    
    def get_dummy_inputs(self):
        U = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]]).view((3,2))
        d = torch.tensor([1.0 ,2.0, 3.0]).view((-1,1))
        x = torch.tensor([4.0, 5.0, 6.0]).view((-1,1))
        A = U @ U.t() + torch.diag(d.flatten())
        I_n = torch.eye(A.shape[0])
        
        return A, U, d, x, I_n
    
    def test_mult(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        self.assertAllClose(lr.mult(U, d, x), A @ x)
        
    def test_invMult(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        self.assertAllClose(lr.invMult(U, d, x), torch.inverse(A) @ x)

    def test_factCore(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        K, _ = lr.factCore(U)
        
        self.assertAllClose(
            (I_n + U @ K @ U.t()) @ (I_n + U @ K @ U.t()).t(), 
            I_n + U @ U.t()
        )
    
    def test_factMult(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        self.assertAllClose(
            lr.factMult(U, d, lr.factMult(U, d, I_n).t()), 
            A
        )
        self.assertEqual(
            lr.factMult(U, d, x).shape, torch.Size([U.shape[0], x.shape[1]])
        )
    
    def test_invFactMult(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        self.assertAllClose(
            lr.invFactMult(U, d, lr.invFactMult(U, d, I_n).t()), 
            torch.inverse(A)
        )
        self.assertEqual(
            lr.invFactMult(U, d, x).shape, torch.Size([U.shape[0], x.shape[1]])
        )
    
    def test_logdet(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        self.assertAllClose(
            lr.logdet(U, d),
            torch.logdet(A)
        )
        
    def test_trace(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        self.assertAllClose(
            lr.trace(U, d),
            torch.trace(A)
        )

    def test_invFacts(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        
        F1, F2, v = lr.invFacts(U, d)

        self.assertAllClose(
            torch.diag(v.flatten()) - F1 @ F2,
            torch.inverse(A)
        )
        
    def test_reduceRank(self):
        U = torch.randn((5, 2))
        V = torch.cat([U, U], dim=1)
        W = lr.reduceRank(V)
        
        print(U, U.shape)
        print(V, V.shape)
        print(W, W.shape)
        
        print(2*(U @ U.t()))
        print(V @ V.t())
        print(W @ W.t())
        
        self.assertAllClose(V @ V.t(), W @ W.t())

    def test_reducedRank_factCore(self):
        U = torch.randn((5, 2))
        V = torch.cat([U, U], dim=1)
        I_n = torch.eye(5)

        K, W = lr.factCore(V)
        B = I_n + W @ K @ W.t()

        print(B @ B.t())
        print(I_n + V @ V.t())
        print((B @ B.t()) - (I_n + V @ V.t()))
        
        
        self.assertAllClose(B @ B.t(), I_n + V @ V.t())
        
        
