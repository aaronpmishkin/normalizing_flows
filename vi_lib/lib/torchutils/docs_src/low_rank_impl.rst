Implementation Details
======================

Implementation details for the ``torchutils.low_rank`` module (`API <low_rank.html>`_).

If a matrix :math:`A` can be expressed as the sum of a low-rank matrix :math:`UU^\top`, where :math:`U` is a :math:`[n \times k]` matrix, :math:`k \ll n`, and a diagonal :math:`[n \times n]` matrix :math:`D`, :math:`A = UU^\top + D`, it is more efficient to do operations involving :math:`A` using :math:`U, D` directly instead of first computing :math:`A`.

* The memory requirements to store :math:`A` is :math:`O(n^2)`, compared to :math:`O(nk)` for :math:`U, D`.
* Most operations that require :math:`O(n^3)` operations using :math:`A` can be implemented in :math:`O(nk^2)` ops. using :math:`U, D`.

The goal of this library is to make it possible to compute operations involving :math:`A` without forming a :math:`[n \times n]` matrix.
This page gives some pointers to the maths used to implement.

.. note::
	It is assumed that :math:`D` is positive definite, i.e., contain only positive entries.


===================================     ===========================================
	Content
===================================================================================
`Multiplication`_ 						:math:`Ax`
`Inversion`_							:math:`A^{-1}x`
`Symmetric Factorization`_	 			:math:`Bx`, where :math:`BB^\top = A`
`Inverse Symmetric Factorization`_	 	:math:`Cx`, where :math:`CC^\top = A^{-1}`
`Determinant`_	 						:math:`\det(A)`
`Trace`_		 						:math:`\text{Tr}(A)`
===================================     ===========================================

The code examples below assume that ``U`` is a ``[n x k]`` and ``d`` is a ``[n x 1]`` Torch.Tensor.

Multiplication
-------------------------

Matrix multiplication only requires an ordering of operation that avoid creating a :math:`[n \times n]` matrix. 
The following ordering only uses the multiplication of :math:`[n \times k]` matrices with vectors.

.. math::
	Ax = (UU^\top + D)x = U(U^\top x) + Dx

Inversion
----------------

Computing the inverse can be done using `Woodbury's matrix identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_.
Setting :math:`V = D^{-1/2}U`, 

.. math:: 
	\begin{array}{rcl}
		A^{-1}x &=& (UU^\top + D)^{-1}x \\
		&=& \left(D^{1/2}(VV^\top + I_n)D^{1/2}\right)^{-1}\\
		&=& D^{1/2}(VV^\top + I_n)^{-1}D^{-1/2}\\
		&=& D^{-1/2}(I_n - V(I_k + V^\top V)^{-1} V^\top) D^{-1/2} x
	\end{array}
	
Careful ordering to avoid matrix-matrix operations leads to an :math:`O(nk^2)` implementation,

.. math:: 
	A^{-1}x = D^{1/2}\left(\left(I_n - V(I_k + V^\top V\right)^{-1} \left(V^\top \left(D^{1/2} x\right)\right)\right).
	
Factorization Helper
--------------------

For the factorization operations, it is useful to have a subroutine to compute factorizations for matrices of the form :math:`I_n + VV^\top`, i.e., :math:`W` such that :math:`WW^\top = I_n + VV^\top`.
This can be done by using Thm 3.1 from `[AOS14] <https://arxiv.org/abs/1405.0223>`_, which yields a square-root for :math:`I_n + VV^\top`, i.e., :math:`W = W^\top`. Setting 

.. math::
	\begin{array}{rcl}
		L &=& \text{Cholesky}(V^\top V),\\
		M &=& \text{Cholesky}(I_k + V^\top V),\\
		K &=& L^{-\top}(M - I_k) L^{-1},\\
		W &=& I_n + VKV^\top.
	\end{array}

Working with :math:`W` directly is impractical, as it is of size :math:`[n \times n]`. 
The function ``__factCore(V)`` (`Source <_modules/low_rank.html#__factCore>`_) returns the intermediate :math:`[k \times k]` matrix :math:`K` instead.

Symmetric Factorization
-------------------------------------------------------------------------------

Given a way to compute a factor :math:`W` for :math:`VV^\top + I_n`, we have that :math:`B = D^{1/2} W` is a symmetric factorization of :math:`A`; 

.. math:: 
	\begin{array}{rcl}
		A &=& (UU^\top + I_n),\\
		&=& D^{1/2}(VV^\top + I_n)D^{1/2},\\
		&=& D^{1/2}WW^\top D^{1/2}.
	\end{array}

We can compute :math:`K = \texttt{__factCore}(V)` to get :math:`W = I_n + VKV^\top`, and careful ordering to avoid matrix-matrix operations leads to an :math:`O(nk^2)` implementation,

.. math:: 
	\begin{array}{rcl}
		Bx &=& D^{1/2} W x,\\
		&=& D^{1/2}(I_n + V K V^\top)x,\\
		&=& D^{1/2}x + V \left(K \left(V^\top x\right)\right),\\
	\end{array}

Inverse Symmetric Factorization
------------------------------------------------------------------------------

As above, but we now need a symmetric factorization for :math:`CC^\top = A^{-1}`.
This can be done by using `Woodbury's matrix identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ to compute the inverse of :math:`W`. Given that :math:`W` is symmetric, we have that 

.. math:: 
	\begin{array}{rcl}
		A^{-1} &=& (UU^\top + I_n),\\
		&=& D^{-1/2}(VV^\top + I_n)^{-1}D^{-1/2},\\
		&=& D^{1/2}W^{-1}W^{-\top}D^{1/2}.
	\end{array}

Using Woodbury's identity on :math:`W` and ordering as to avoid matrix-matrix multiplication gives an :math:`O(nk^2)` algorithm,
	
.. math:: 
	\begin{array}{rcl}
		Cx &=& D^{1/2}W^{-1}x,\\
		&=& D^{1/2}(I_n + V K V^\top)^{-1}x,\\
		&=& D^{1/2}(I_n - V (K^{-1} + V^\top V)^{-1} V^\top)x,\\
		&=& D^{1/2}x - V \left((K^{-1} + V^\top V)^{-1} \left(V^\top x\right)\right).\\
	\end{array}

Determinant
-------------------------------------------------------------------------------

Using the multiplicity of the determinant, we have that 

.. math::
	\begin{array}{rcl}
		\det(A) &=& \det(UU^\top + D),\\
		&=& \det(D^{1/2}(VV^\top + I_n)D^{1/2}),\\
		&=& \det(VV^\top + I_n)\det(D).\\
	\end{array}

Using `Sylvester's Determinant Identity <https://en.wikipedia.org/wiki/Sylvester's_determinant_identity>`_, we have that :math:`\det(VV^\top + I_n) = \det(V^\top V + I_k)`.

The determinant involves computing large products, :math:`\det(D) = \prod_{i=1}^n D_{ii}`, which can be unstable if :math:`n` is large. For stability, we return the log-determinant,

.. math::
	\begin{array}{rcl}
		\log \det(A) &=& \det(VV^\top + I_n)\det(D),\\
		&=& \log \det(V^\top V + I_k) + \sum_{i=1}^n D_{ii}.
	\end{array}

Trace
-------------------------------------------------------------------------------

The trace is not difficult to compute given :math:`A`, as it is simply the sum of the elements on the diagonal - the challenge is in avoiding the computation of :math:`A`.
As 

.. math::

	(UU^\top)_{dd} = \sum_{i=1}^k (U)_{di} (U^\top)_{id} = \sum_{i=1}^k U_{di}^2,

we can get the sum of the elements on the diagonal of :math:`UU^T` by taking ``torch.sum(U**2)``, leading to a simple 

.. code-block:: python 

	def trace(U, d): return torch.sum(d) + torch.sum(U**2)



