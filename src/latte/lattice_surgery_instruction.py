import dataclasses
from typing import Literal, Any, Optional

TNode = Any


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class ErrorSource:
    error_type: str
    error_basis: Literal['X', 'Z']
    error_location: complex
    error_initiative: Literal['before', 'after', 'during']
    error_layer: Any


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class LatticeSurgeryInstruction:
    """An instruction used when simulating lattice surgery."""

    # The instruction to perform.
    action: Literal[
        # Allocate a new qubit and initialize it to the |+> state.
        'qalloc_x',
        # Allocate a new qubit and initialize it to the |i> state.
        'qalloc_y',
        # Allocate a new qubit and initialize it to the |0> state.
        'qalloc_z',

        # Measure a qubit in the X basis then discard the qubit.
        'm_discard_x',
        # Measure a qubit in the Y basis then discard the qubit.
        'm_discard_y',
        # Measure a qubit in the Z basis then discard the qubit.
        'm_discard_z',

        # Measure the X-basis parity of two qubits.
        'mxx',
        # Measure the Z-basis parity of two qubits.
        'mzz',

        # Perform a Pauli X gate.
        'x',
        # Perform a Pauli Y gate.
        'y',
        # Perform a Pauli Z gate.
        'z',
        # Perform a CNOT gate controlled by target flipping target2.
        'cx',
        # Perform a Hadamard gate.
        'h',
        # Perform a T gate.
        't',
        # Perform an S gate.
        's',

        # Randomly apply an X but return a measurement result indicating if it happened.
        'heralded_random_x',
        # Randomly apply a Z but return a measurement result indicating if it happened.
        'heralded_random_z',

        # Apply an X gate controlled by a measurement result.
        'feedback_m2x',
        # Apply a Y gate controlled by a measurement result.
        'feedback_m2y',
        # Apply a Z gate controlled by a measurement result.
        'feedback_m2z',

        # Clear the accumulator bit in preparation for the next compound measurement.
        'accumulator_bit_clear',
        # Xor a measurement result into the accumulator bit.
        'accumulator_bit_xor',
        # Save the accumulator bit as a measurement result.
        'accumulator_bit_save',

        # An X gate that will be performed or not, based on external configuration.
        'error_mechanism_x',
        # A Y gate that will be performed or not, based on external configuration.
        'error_mechanism_y',
        # A Z gate that will be performed or not, based on external configuration.
        'error_mechanism_z',
        # A measurement error to queue to perform or not, based on external configuration.
        'error_mechanism_m',
    ]
    # The qubit to target with the gate. None if the gate only operates on measurements.
    target: Any = None
    # The second qubit to target with the gate. None if the gate targets no qubits or 1 qubit.
    target2: Any = None
    # Where to store a measurement result, or where to read a measurement result from.
    measure_key: Any = None
    # If this is an error mechanism, where did that error come from.
    error_source: Optional[ErrorSource] = None

    def __str__(self) -> str:
        terms = [self.action]
        for x in [self.target, self.target2]:
            if x is None:
                continue
            r = x.real
            i = x.imag
            if r == int(r):
                r = int(r)
            if i == int(i):
                i = int(i)
            if i == 0 and r == 0:
                terms.append('0')
            elif r == 0:
                terms.append(f'{i}i')
            elif i == 0:
                terms.append(f'{r}')
            elif i < 0:
                terms.append(f'{r}{i}i')
            else:
                terms.append(f'{r}+{i}i')
        if self.measure_key:
            terms.append(repr(self.measure_key))
        return ' '.join(terms)
