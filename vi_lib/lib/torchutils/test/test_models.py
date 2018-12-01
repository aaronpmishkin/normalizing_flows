import unittest
import torch 
import torchutils.models as models

from torchutils.params import bp2v
from torch.nn.utils import vector_to_parameters as v2p
from torch.nn.utils import parameters_to_vector as p2v

class MLPTestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b, 0.01))
    
    def get_dummy_inputs(self, n, indim, hiddim, outdim, s):
        torch.manual_seed(0)

        mlp = models.MLP(indim, hiddim, outdim)
        x = torch.rand(n, indim)

        noise = torch.randn(s, models.num_params(mlp))
        
        return mlp, x, noise

    def test_num_params(self):
        self.assertEqual(models.num_params(models.MLP(10,[],1)), (10+1))
        self.assertEqual(models.num_params(models.MLP(10,[1],1)), (10+1) + (1+1))
        self.assertEqual(models.num_params(models.MLP(10,[2],1)), (10+1)*2 + (2+1))

    def test_interface_forward(self): 
        mlp, x, _, = self.get_dummy_inputs(7, 5, [], 1, 3)

        y = mlp(x)

        self.assertTrue(y.shape[0] == x.shape[0])
        self.assertTrue(y.shape[1] == 1)

    def test_interface_forward_with_noise(self):
        n, s = 7, 3

        mlp, x, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        print(list(mlp.parameters()))
        y = mlp(x, noise)
        self.assertTrue(list(y.shape) == [s, n, 1])

        mlp, x, noise = self.get_dummy_inputs(n, 5, [11], 1, s)
        y = mlp(x, noise)
        self.assertTrue(list(y.shape) == [s, n, 1])

    def test_backward_with_noise(self):
        n, s = 7, 3

        def manual_gradient(mlp, x, noise):
            mu = p2v(mlp.parameters())
            
            gs = []
            for sid in range(s):
                v2p((noise[sid,:] + mu).contiguous(), mlp.parameters())
                g = torch.autograd.grad(torch.sum(mlp(x)), mlp.parameters())
                print([gg.shape for gg in g])
                gs.append(bp2v(g, 0))

            v2p(mu, mlp.parameters())

            return sum(gs)

        mlp, x, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        grad1 = p2v(torch.autograd.grad(torch.sum(mlp(x, noise)), mlp.parameters()))
        grad2 = manual_gradient(mlp, x, noise)
        self.assertAllClose(grad1, grad2)

        mlp, x, noise = self.get_dummy_inputs(n, 5, [11], 1, s)

        grad1 = p2v(torch.autograd.grad(torch.sum(mlp(x, noise)), mlp.parameters()))
        grad2 = manual_gradient(mlp, x, noise)
        self.assertAllClose(grad1, grad2)
