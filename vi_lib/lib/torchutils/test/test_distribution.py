
import unittest
import torch 
import torchutils.distributions as distr

kl_ref = torch.distributions.kl.kl_divergence

def kl_ref2(a, A, b, B):
    r"""KL(N(a, A), N(b, B))"""
    if False:
        return 0.5*(
                torch.trace(torch.inverse(B) @ A)
                + (b-a).t() @ torch.inverse(B) @ (b-a)
                - torch.numel(a)
                + torch.logdet(B)
                - torch.logdet(A)
            )
    else:
        return [
            (b-a).t() @ torch.inverse(B) @ (b-a),
            torch.logdet(B),
            - torch.logdet(A),
            torch.numel(a),
            torch.trace(torch.inverse(B) @ A),
            ]


class LowRankMultivariateNormalTestCase(unittest.TestCase):

    def assertAllClose(self, a, b):
        self.assertTrue(torch.allclose(a, b, 0.01), [a, b])
    
    def get_dummy_inputs(self):
        torch.manual_seed(0)
        U = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]]).view((3,2))
        d = torch.tensor([1.0 ,2.0, 3.0]).view((-1,1))
        x = torch.tensor([4.0, 5.0, 6.0]).view((-1,1))
        A = U @ U.t() + torch.diag(d.flatten())
        I_n = torch.eye(A.shape[0])
        
        return A, U, d, x, I_n
    
    def test_rsample(self):
        A, U, d, x, I_n = self.get_dummy_inputs()
        N = 1000000
        
        p = distr.LowRankMultivariateNormal(torch.zeros_like(x), cov_factor=U, cov_diag=d)
        X = p.rsample(n_samples=N)
        self.assertAllClose((X @ X.t())/N, A)
        
        p = distr.LowRankMultivariateNormal(torch.zeros_like(x), prec_factor=U, prec_diag=d)
        X = p.rsample(n_samples=N)
        self.assertAllClose((X @ X.t())/N, torch.inverse(A))

    def kl_dummy_inputs(self):
        torch.manual_seed(0)
        mu1 = torch.randn(3,1)
        mu2 = torch.randn(3,1)
        U = torch.randn(3,2)
        V = torch.randn(3,2)
        d = torch.randn(3,1)**2
        b = torch.randn(3,1)**2
        A = U @ U.t() + torch.diag(d.flatten())
        B = V @ V.t() + torch.diag(b.flatten())
        return mu1, U, d, mu2, V, b, A, B
    
    def test_kl_lrmnC_lrmnP(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()

        p = distr.LowRankMultivariateNormal(loc=mu1, cov_factor=U, cov_diag=d)
        q = distr.LowRankMultivariateNormal(loc=mu2, prec_factor=V, prec_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), covariance_matrix=A)
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), precision_matrix=B)

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))

    def test_kl_lrmnC_lrmnC(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()
        
        p = distr.LowRankMultivariateNormal(loc=mu1, cov_factor=U, cov_diag=d)
        q = distr.LowRankMultivariateNormal(loc=mu2, cov_factor=V, cov_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), covariance_matrix=A)
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), covariance_matrix=B)

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))

    def test_kl_lrmnP_lrmnP(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()

        p = distr.LowRankMultivariateNormal(loc=mu1, prec_factor=U, prec_diag=d)
        q = distr.LowRankMultivariateNormal(loc=mu2, prec_factor=V, prec_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), precision_matrix=A)
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), precision_matrix=B)

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))

    def test_kl_lrmnP_mfmnP(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()
        
        p = distr.LowRankMultivariateNormal(loc=mu1, prec_factor=U, prec_diag=d)
        q = distr.MeanFieldMultivariateNormal(loc=mu2, prec_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), precision_matrix=A)
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), precision_matrix=torch.diag(b.flatten()))

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))

    def test_kl_lrmnC_mfmnP(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()

        p = distr.LowRankMultivariateNormal(loc=mu1, cov_factor=U, cov_diag=d)
        q = distr.MeanFieldMultivariateNormal(loc=mu2, prec_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), covariance_matrix=A)
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), precision_matrix=torch.diag(b.flatten()))

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))

    def test_kl_mfmnP_mfmnO(self):
        mu1, U, d, mu2, V, b, A, B = self.kl_dummy_inputs()

        p = distr.MeanFieldMultivariateNormal(loc=mu1, prec_diag=d)
        q = distr.MeanFieldMultivariateNormal(loc=mu2, prec_diag=b)
        P = torch.distributions.multivariate_normal.MultivariateNormal(mu1.flatten(), precision_matrix=torch.diag(d.flatten()))
        Q = torch.distributions.multivariate_normal.MultivariateNormal(mu2.flatten(), precision_matrix=torch.diag(b.flatten()))

        self.assertAllClose(p.kl(q), kl_ref(P, Q))
        self.assertAllClose(q.kl(p), kl_ref(Q, P))
