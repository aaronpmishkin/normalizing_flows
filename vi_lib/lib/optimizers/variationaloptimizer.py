r"""
Base class for variational optimization
"""
import lib
from lib.optimizers.optimizer import Optimizer

class VariationalOptimizer(Optimizer):
    r"""Base class for variational optimization procedure

    .. todo::

        document the requirements for the closure
    """

    def step(self, closure):
        r"""
        .. todo::

            document the requirements for the closure
        """
        raise NotImplementedError

    def distribution(self):
        r"""Returns a probability distribution over the parameters
        """
        raise NotImplementedError
