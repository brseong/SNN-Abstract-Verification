from z3.z3 import Int, Real, RealVal, Solver, And, Implies, sat, ArithRef, BoolRef
from uuid import uuid4
from ..types import Z3Data, Abstraction
import torch as th


def floor(in_: ArithRef, floor_: ArithRef) -> BoolRef:
    """Define the floor relation.

    Args:
        in_ (ArithRef): The input value of floor function, a real number
        floor_ (ArithRef): The floor value of the input, an integer

    Returns:
        BoolRef: Z3 expression representing the floor relation. True if the floor_ is the floor of in_, False otherwise
    """
    return And(floor_ <= in_, in_ < floor_ + 1) # type: ignore

def clamp(in_: ArithRef, min_: int, max_: int) -> BoolRef:
    return And(min_ <= in_, in_ <= max_) # type: ignore

def generate_snn(s: Solver, weight:list[th.Tensor], data: Z3Data):
    """Generate SMT encoding for the SNN

    Args:
        s (Solver): Solver object
        weight (list[th.Tensor]): List of weight tensors. Each tensor is of shape (n_features[layer], n_features[layer + 1])
        data (Z3Data): Z3Data object, used to store data throughout the experiment
    """
    # Generate integer variables for the number of spikes
    for postsynaptic_layer in range(len(data.n_features)):
            for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
                data.n_spikes[postsynaptic_layer, postsynaptic_neuron] = Int(
                    f"n_spks_{postsynaptic_layer}_{postsynaptic_neuron}"
                )
        
    # Generate real variables for the weights
    for presynaptic_layer, postsynaptic_layer in zip(range(len(data.n_features) - 1), range(1, len(data.n_features))):
        for presynaptic_neuron in range(data.n_features[presynaptic_layer]):
            for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
                data.weight[presynaptic_layer, presynaptic_neuron, postsynaptic_neuron] = weight[presynaptic_layer][presynaptic_neuron, postsynaptic_neuron].item()
                
    for postsynaptic_layer in range(1, len(data.n_features)):
        for postsynaptic_neuron in range(data.n_features[postsynaptic_layer]):
            presynaptic_layer = postsynaptic_neuron - 1
            epsp = 0
            for presynaptic_neuron in range(data.n_features[presynaptic_layer]):
                epsp += data.weight[presynaptic_layer, presynaptic_neuron, postsynaptic_neuron] * data.n_spikes[presynaptic_layer, presynaptic_neuron]
                s.add(
                    floor(epsp, data.n_spikes[postsynaptic_layer, postsynaptic_neuron])
                )
            del epsp
    
    prediction = Int("prediction")
    s.add(clamp(prediction, 0, data.n_features[-1] - 1))
    for candidate in range(data.n_features[-1]):
        sub_terms_ = []
        for other in range(data.n_features[-1]):
            if candidate != other:
                sub_terms_.append(data.n_spikes[-1, candidate] > data.n_spikes[-1, other])
        s.add(Implies(prediction == candidate, And(sub_terms_)))
        del sub_terms_
    
    return s

def generate_adversarial_constraints(s: Solver, data: Z3Data, abstraction: Abstraction):
    # Clamping the number of spikes
    for layer in range(len(data.n_features)):
        for step in range(data.n_steps):
            for neuron in range(data.n_features[layer]):
                s.add(clamp(data.n_spikes[layer, neuron], 0, 1))
    
    
