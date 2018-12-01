TorchUtils
==========

This package provides some utility functions based on PyTorch.

`GitHub <https://github.com/fkunstner/torchutils>`_ - `Documentation <https://fkunstner.github.io/torchutils/>`_

You might also be interested in checking out 

`Inferno <https://github.com/inferno-pytorch/inferno>`_, 
`ProbTorch <https://github.com/probtorch/probtorch>`_, 
`Pyro <https://github.com/uber/pyro>`_, 
`Ignite <https://github.com/pytorch/ignite>`_

Installation
------------

Installing from git
^^^^^^^^^^^^^^^^^^^ 

Editable mode: 

.. code-block:: guess
	
	git clone https://github.com/fKunstner/torchutils.git torchutils
	cd torchutils
	pip install -e .

Direct install:

.. code-block:: guess

	pip install git+https://github.com/fKunstner/torchutils.git

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

Currently disabled 

.. code-block:: guess 
	
	pip install fk-torchutils 
	
Modules
-------
* ``torchutils.low_rank`` - `Fast Operations for Low-Rank + Diagonal Matrices <low_rank.html>`_
* ``torchutils.distributions`` - `Low-Rank + Diagonal Multivariate Gaussian <distributions.html>`_
* ``torchutils.models`` - `(very) Simple models instantiation helpers <models.html>`_
* ``torchutils.fastpca`` - `Fast Randomized PCA <fastpca.html>`_
