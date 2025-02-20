from z3.z3 import Int, Solver, And, Implies, sat, ArithRef
from uuid import uuid4
from .types import Z3Data


def floor(in_: ArithRef, floor_: ArithRef):
    return And(floor_ <= in_, in_ < floor_ + 1)


def generate_snn(s: Solver, data: Z3Data):
    for layer in range(3):
        for step in range(50):
            for neuron in range(10):
                data.n_spikes[layer, step, neuron] = Int(
                    f"n_spks_{layer}_{step}_{neuron}"
                )
