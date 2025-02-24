from z3.z3 import Int, Real, RealVal, Solver, And, Implies, sat, ArithRef, BoolRef
from uuid import uuid4
from ..types import Z3Data, Abstraction
import torch as th


def floor(s: Solver, _in: ArithRef, _floor: ArithRef) -> Solver:
    """Define the floor relation. True if the _floor is the floor of _in, False otherwise

    Args:
        s (Solver): Solver object
        _in (ArithRef): The input value of floor function, a real number
        _floor (ArithRef): The floor value of the input, an integer

    Returns:
        Solver: Solver object with the floor relation added
    """
    return s.add(And(_floor <= _in, _in < _floor + 1))  # type: ignore


def clamp(s: Solver, _in: ArithRef, _min: int, _max: int) -> Solver:
    """Clamp the input value between _min and _max

    Args:
        s (Solver): Solver object
        _in (ArithRef): The input value to be clamped
        _min (int): The minimum value of the clamping
        _max (int): The maximum value of the clamping

    Returns:
        Solver: Solver object with the clamping relation added
    """
    s.add(And(_min <= _in, _in <= _max))  # type: ignore
    return s


def generate_snn(s: Solver, weight: list[th.Tensor], data: Z3Data):
    """Generate SMT encoding for the SNN

    Args:
        s (Solver): Solver object
        weight (list[th.Tensor]): List of weight tensors. Each tensor is of shape (n_features[layer], n_features[layer + 1])
        data (Z3Data): Z3Data object, used to store data throughout the experiment

    Returns:
        Solver: Solver object with the SMT encoding of the SNN
    """
    # Generate integer variables for the number of spikes
    for postsynaptic_layer in range(len(data.n_features)):
        for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
            data.n_spikes[postsynaptic_layer, postsynaptic_neuron] = Int(
                f"n_spks_{postsynaptic_layer}_{postsynaptic_neuron}"
            )

    # Generate real variables for the weights
    for presynaptic_layer, postsynaptic_layer in zip(
        range(len(data.n_features) - 1), range(1, len(data.n_features))
    ):
        for presynaptic_neuron in range(data.n_features[presynaptic_layer]):
            for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
                data.weight[
                    presynaptic_layer, presynaptic_neuron, postsynaptic_neuron
                ] = weight[presynaptic_layer][
                    presynaptic_neuron, postsynaptic_neuron
                ].item()

    for postsynaptic_layer in range(1, len(data.n_features)):
        for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
            presynaptic_layer = postsynaptic_neuron - 1
            epsp = 0
            for presynaptic_neuron in range(data.n_features[presynaptic_layer]):
                epsp += (
                    data.weight[
                        presynaptic_layer, presynaptic_neuron, postsynaptic_neuron
                    ]
                    * data.n_spikes[presynaptic_layer, presynaptic_neuron]
                )
                floor(s, epsp, data.n_spikes[postsynaptic_layer, postsynaptic_neuron])
            del epsp

    prediction = Int("prediction")
    clamp(s, prediction, 0, data.n_features[-1] - 1)
    for candidate in range(data.n_features[-1]):
        sub_terms_ = []
        for other in range(data.n_features[-1]):
            if candidate != other:
                sub_terms_.append(
                    data.n_spikes[-1, candidate] > data.n_spikes[-1, other]
                )
        s.add(Implies(prediction == candidate, And(sub_terms_)))
        del sub_terms_

    return s


def allocate_input(s: Solver, data: Z3Data, _input: th.Tensor) -> Solver:
    """Allocate the input values to the SNN

    Args:
        s (Solver): Solver object
        data (Z3Data): Z3Data object
        _input (th.Tensor): Input tensor, shape (n_timesteps, n_features[0])

    Returns:
        Solver: Solver object with the input values allocated
    """
    input_spikes = _input.sum(dim=0)
    for neuron in range(data.n_features[0]):
        s.add(data.n_spikes[0, neuron] == input_spikes[neuron].item())

    return s


def generate_adversarial_constraints(s: Solver, data: Z3Data, abstraction: Abstraction):
    # Clamping the number of spikes
    for layer in range(len(data.n_features)):
        for step in range(data.n_steps):
            for neuron in range(data.n_features[layer]):
                s.add(clamp(data.n_spikes[layer, neuron], 0, 1))
