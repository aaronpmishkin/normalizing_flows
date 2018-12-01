Fast Randomized PCA
================================================================

This module provides a PyTorch implementation of fast randomized top-k eigendecomposition and borrows heavily from Facebook's ``fbpca`` code (basically an incomplete port from Numpy to Pytorch).

See their `Github <https://github.com/facebook/fbpca>`_ | `Doc <http://fbpca.readthedocs.io/en/latest/>`_ | `Blog post <https://research.fb.com/fast-randomized-svd/>`_.


.. autosummary::
   :nosignatures:

   fastpca.eigsh
   fastpca.eigsh_func
   
.. autofunction:: fastpca.eigsh
.. autofunction:: fastpca.eigsh_func
