import pdb
import torch as th
from z3.z3 import (
    Int,
    Real,
    RealVal,
    Solver,
    And,
    Implies,
    sat,
    ArithRef,
    BoolRef,
    If,
    Sum,
)
from uuid import uuid4
from ..types import Z3Data, Abstraction
from tqdm.auto import tqdm


def floor(s: Solver, _in: ArithRef, _floor: ArithRef) -> Solver:
    """Define the floor relation. ``True`` if the ``_floor`` is the floor of ``_in``, ``False`` otherwise

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


def generate_snn(
    s: Solver,
    weight_list: list[th.Tensor],
    data: Z3Data,
    save_path: str | None = None,
    load_sexpr: bool = False,
) -> Solver:
    """Generate SMT encoding for the SNN.
    Args:
        s (Solver): Solver object
        weight_list (list[th.Tensor]): List of weight tensors. Each tensor is of shape (``n_features[layer + 1]``, ``n_features[layer]``)
        data (Z3Data): Z3Data object, used to store data throughout the experiment

    Returns:
        Solver: Solver object with the SMT encoding of the SNN
    """
    # Generate integer variables for the number of spikes
    for postsynaptic_layer in tqdm(
        range(len(data.n_features)), desc="Generating spike counts..."
    ):
        for postsynaptic_neuron in tqdm(
            range(data.n_features[postsynaptic_layer]), leave=False
        ):
            data.n_spikes[postsynaptic_layer, postsynaptic_neuron] = Int(
                f"#_of_spks_{postsynaptic_layer}_{postsynaptic_neuron}"
            )

    # Generate real variables for the weights
    for presynaptic_layer, postsynaptic_layer in tqdm(
        zip(range(len(data.n_features) - 1), range(1, len(data.n_features))),
        desc="Generating weights...",
    ):
        for presynaptic_neuron in tqdm(
            range(data.n_features[presynaptic_layer]), leave=False
        ):
            for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
                data.weight[
                    presynaptic_layer, presynaptic_neuron, postsynaptic_neuron
                ] = weight_list[presynaptic_layer][
                    postsynaptic_neuron, presynaptic_neuron
                ].item()

    if load_sexpr:
        if save_path is None:
            raise ValueError("save_path must be provided if load_sexpr is True")
        with open(save_path, "r") as f:
            s.from_string(f.read())
        return s

    # Describe the dynamics of the SNN
    for postsynaptic_layer in tqdm(
        range(1, len(data.n_features)), desc="Generating SNN dynamics..."
    ):
        for postsynaptic_neuron in tqdm(
            range(data.n_features[postsynaptic_layer]), leave=False
        ):
            presynaptic_layer = postsynaptic_layer - 1
            epsps = list[ArithRef]()
            for presynaptic_neuron in range(data.n_features[presynaptic_layer]):
                epsps.append(
                    data.weight[
                        presynaptic_layer, presynaptic_neuron, postsynaptic_neuron
                    ]
                    * data.n_spikes[presynaptic_layer, presynaptic_neuron]
                )
            epsp_sum = Sum(epsps)
            clamped_epsp = Real(
                f"clamped_epsp_{postsynaptic_layer}_{postsynaptic_neuron}"
            )
            s.add(clamped_epsp == If(epsp_sum > 0, 1, epsp_sum))
            s.add(
                Implies(
                    clamped_epsp < 0,
                    data.n_spikes[postsynaptic_layer, postsynaptic_neuron] == 0,
                )
            )  # type: ignore
            s.add(
                Implies(
                    clamped_epsp >= 0,
                    And(
                        data.n_spikes[postsynaptic_layer, postsynaptic_neuron]
                        <= clamped_epsp,
                        clamped_epsp
                        < data.n_spikes[postsynaptic_layer, postsynaptic_neuron] + 1,
                    ),
                )
            )
            del epsp

    # Describe the prediction of the SNN
    prediction = Int("prediction")
    clamp(s, prediction, 0, data.n_features[-1] - 1)
    for candidate in range(data.n_features[-1]):
        sub_terms = list[BoolRef]()
        for other in range(data.n_features[-1]):
            if candidate != other:
                sub_terms.append(
                    data.n_spikes[len(data.n_features) - 1, candidate]
                    > data.n_spikes[len(data.n_features) - 1, other]
                )
        s.add(Implies(prediction == candidate, And(sub_terms)))  # type: ignore
        del sub_terms

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(s.sexpr())
            print(f"Saved SMT encoding to {save_path}")

    return s


def allocate_input(s: Solver, data: Z3Data, _input: th.Tensor) -> Solver:
    """Allocate the input values to the SNN

    Args:
        s (Solver): Solver object
        data (Z3Data): Z3Data object
        _input (th.Tensor): Input tensor, shape (``n_steps``, ``n_features[0]``)

    Returns:
        Solver: Solver object with the input values allocated
    """
    input_spikes = _input.sum(dim=0)
    for neuron in range(data.n_features[0]):
        s.add(data.n_spikes[0, neuron] == input_spikes[neuron].item())  # type: ignore

    return s


def generate_adversarial_constraints(s: Solver, data: Z3Data, abstraction: Abstraction):
    # Clamping the number of spikes
    for layer in range(len(data.n_features)):
        for step in range(data.n_steps):
            for neuron in range(data.n_features[layer]):
                pass
