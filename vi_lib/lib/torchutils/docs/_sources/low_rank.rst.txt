Fast Operations with Low-Rank + Diagonal Matrices
================================================================

.. automodule:: low_rank

The original goal was to implement operations to sample and update from gaussian distributions with low-rank + diagonal covariance/precision matrices, which has since been implemented in PyTorch (see this `PR <https://github.com/pytorch/pytorch/pull/8635>`_).

More background is available at the `implementation details <low_rank_impl.html>`_ page if you want to implement more operations.

.. autosummary::

   low_rank.mult
   low_rank.invMult
   low_rank.factMult
   low_rank.invFactMult
   low_rank.logdet
   low_rank.trace
   
.. note:: General 
	The matrix :math:`U` is assumed to be of size :math:`[n \times k]`, where :math:`k \ll n`.
   
.. autofunction:: low_rank.mult
.. autofunction:: low_rank.invMult
.. autofunction:: low_rank.factMult
.. autofunction:: low_rank.invFactMult
.. autofunction:: low_rank.logdet
.. autofunction:: low_rank.trace

.. toctree::
	:caption: Content
	:hidden:
	:maxdepth: 2

	low_rank_impl
