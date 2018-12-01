import unittest
import torch 
import torchutils as tu 
import torchutils.models as models
from torchutils.params import bv2p, bp2v

class ParamsTestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b, 0.01))
    
    def testSomething(self):
        N, D, S, H = 3, 5, 7, [11]
        
        mlp = models.MLP(D, H, 1)
        x = torch.randn(N, D)
        
        y = mlp(x)
        print("yshape", y.shape)
        
    
        noise = torch.randn(S, tu.params.num_params(mlp.parameters()))
        print("noiseshape", noise.shape)
        y, a, b = mlp(x, noise, indgrad=True) 
        print("yshape2", y.shape)
        
        f = tu.curvfuncs.closure_factory(mlp, x, lambda x : torch.norm(x), ["grad", "grads"])
        res = f()
        print(res)
        
        #self.assertTrue(False)