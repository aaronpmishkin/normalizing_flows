import unittest
import torch 
import torchutils.distributions as distr
import torchutils.curvfuncs as closures
import torchutils.models as models 

from torch.nn.utils import vector_to_parameters as v2p
from torch.nn.utils import parameters_to_vector as p2v

class CurvFuncsTestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b, 0.01))
    
    def get_dummy_inputs(self, n, indim, hiddim, outdim, s):
        torch.manual_seed(0)

        mlp = models.MLP(indim, hiddim, outdim)
        x = torch.rand(n, indim)
        y = torch.rand(n, 1)
        loss = lambda x : .5*torch.norm(x-y)**2
        noise = torch.randn(s, models.num_params(mlp))
        
        return mlp, x, loss, noise

    def test_grad1(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        g1 = torch.autograd.grad(loss(mlp(x)), mlp.parameters())
        _, g2 = closures.closure_factory(mlp, x, loss, ['grad_pf'])()
        _, g3 = closures.closure_factory(mlp, x, loss, ['grad'])()
        self.assertAllClose(p2v(g1), p2v(g2))
        self.assertAllClose(p2v(g1), g3)

    def test_grad2(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [11], 1, s)
        g1 = torch.autograd.grad(loss(mlp(x)), mlp.parameters())
        _, g2 = closures.closure_factory(mlp, x, loss, ['grad_pf'])()
        _, g3 = closures.closure_factory(mlp, x, loss, ['grad'])()
        
        self.assertAllClose(p2v(g1), p2v(g2))
        self.assertAllClose(p2v(g1), g3)

    def test_grad3(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        g1 = torch.autograd.grad(loss(mlp(x, noise)), mlp.parameters())
        _, g2 = closures.closure_factory(mlp, x, loss, ['grad_pf'])(noise)
        _, g3 = closures.closure_factory(mlp, x, loss, ['grad'])(noise)
        self.assertAllClose(p2v(g1), p2v(g2))
        self.assertAllClose(p2v(g1), g3)

    def test_grad4(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [11], 1, s)
        g1 = torch.autograd.grad(loss(mlp(x, noise)), mlp.parameters())
        _, g2 = closures.closure_factory(mlp, x, loss, ['grad_pf'])(noise)
        _, g3 = closures.closure_factory(mlp, x, loss, ['grad'])(noise)
        self.assertAllClose(p2v(g1), p2v(g2))
        self.assertAllClose(p2v(g1), g3)

    def test_grads(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        d = models.num_params(mlp)
        _, g = closures.closure_factory(mlp, x, loss, ['grads'])()
        self.assertEqual([n,d], list(g.shape))

    def test_grads2(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [11], 1, s)
        d = models.num_params(mlp)
        _, g = closures.closure_factory(mlp, x, loss, ['grads'])()
        self.assertEqual([n,d], list(g.shape))

    def test_grads3(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [], 1, s)
        d = models.num_params(mlp)
        _, g = closures.closure_factory(mlp, x, loss, ['grads'])(noise)
        self.assertEqual([s,n,d], list(g.shape))

    def test_grads4(self):
        n,s  = 3, 7
        mlp, x, loss, noise = self.get_dummy_inputs(n, 5, [11, 13], 1, s)
        d = models.num_params(mlp)
        _, g = closures.closure_factory(mlp, x, loss, ['grads'])(noise)
        self.assertEqual([s,n,d], list(g.shape))
