import unittest
import torch 
import torchutils.fastpca as fastpca

class FastPCATestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b), str(a) + str(b))
    
    def get_dummy_inputs(self):
        U = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]]).view((3,2))
        d = torch.tensor([1.0 ,2.0, 3.0]).view((-1,1))
        x = torch.tensor([4.0, 5.0, 6.0]).view((-1,1))
        A = U @ U.t() + torch.diag(d.flatten())
        I_n = torch.eye(A.shape[0])
        
        return A, U, d, x, I_n
    
    def test_eigsh(self):
        A, _, _, _, _ = self.get_dummy_inputs()
        
        e, v = fastpca.eigsh(A, k=1)
        E, V = torch.eig(A, eigenvectors=True)

        self.assertAllClose(e[0], E[0,0])
        self.assertAllClose(v[:,0], V[:,0])

    def test_eigsh_func(self):
        A, _, _, _, _ = self.get_dummy_inputs()

        e, v = fastpca.eigsh_func(
            lambda x: A @ x, A.dtype, A.device, A.shape[0], 
            k=1
        )
        E, V = torch.eig(A, eigenvectors=True)

        self.assertAllClose(e[0], E[0,0])
        self.assertAllClose(v[:,0], V[:,0])

    def test_reconstruction_error(self):
        D = 10
        k = 5
        
        A = torch.randn(D, D)
        
        e, v = fastpca.eigsh_func(
            lambda x: A @ A.t() @ x, A.dtype, A.device, D, 
            k=k
        )
        
        E, V = torch.symeig(A @ A.t(), eigenvectors=True)
        (_, idx) = torch.abs(E).sort()
        idx = idx[(D - k):]
        E, V = E[idx], V[:, idx]

        diff = torch.norm(
                (V @ torch.diag(E) @ V.t()) -
                (v @ torch.diag(e) @ v.t())
            )
        self.assertTrue((diff < e-4).all())
