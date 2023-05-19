from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    ls_forward = []  # list containing forward difference
    ls_backward = []  # list containing backward difference

    x_i = vals[arg]
    x_last = x_i - epsilon
    x_next = x_i + epsilon

    for i in range(len(vals)):
        # only camputes an approximation to the derivative with respect to one arg
        if i == arg:
            ls_forward.append(x_next)
            ls_backward.append(x_last)
        # leave the original vlue unchanged
        else:
            ls_forward.append(vals[i])
            ls_backward.append(vals[i])

    # Turn 2-argument function into 1-arg from modul 1.1 page 36
    f_forward = f(*tuple(ls_forward))
    f_backward = f(*tuple(ls_backward))

    # definition of central difference from module 1.1 page 32~33
    f_prime = (f_forward - f_backward) / (2 * epsilon)
    return f_prime

    # alternative:
    # arg_1 = [i for i in vals]
    # arg_1[arg] += epsilon
    # m = f(*arg_1)
    # arg_1[arg] -= 2 * epsilon
    # n = f (*arg_1)
    # return (m-n)/(2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # reference: https://en.wikipedia.org/wiki/Topological_sorting

    # list to contain the sorted nodes
    order: List[Variable] = []

    # use set() method to convert any of the iterable to sequence of iterable elements with distinct elements
    visited = set()

    def visit(n: Variable) -> None:
        # pass if visited before
        if n.unique_id in visited:
            return
        # if not leaf, then visit
        if not n.is_leaf():
            for i in n.parents:
                if not i.is_constant():
                    visit(i)

        # add current nonde into visited set
        visited.add(n.unique_id)
        # add current node into the front of order list
        order.insert(0, n)

    # call visit function
    visit(variable)
    return order
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
    else:
        for (v_d, deriv) in variable.chain_rule(deriv):
            backpropagate(v_d, deriv)
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
