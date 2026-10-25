# """
# NEURON PROJECT
# TensorFlow/Keras rewrite  --> claude anthropic

# Biologically-inspired neuron layer using receptor interaction rules.
# Each neuron maintains a trainable Q-factor per receptor that scales
# stimulatory and inhibitory signals independently.

# Original concept: T. Kowalski
# TF rewrite: fixes all bugs from the numpy prototype and maps the
#             design onto a standard Keras Layer + training loop.
# """

# import numpy as np
# import tensorflow as tf
# import pickle
"""
NEURON PROJECT
TensorFlow/Keras implementation of a biologically inspired receptor layer.

Important modelling note:
- RECEPTOR_RULES describes a simplified, user-defined interaction graph.
- Connections are model assumptions, not a validated biological interaction database.
- Inputs have shape (batch, num_neurons, N_RECEPTORS).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Receptor interaction rules
# Each target receptor lists source receptors that stimulate or inhibit it.
# ---------------------------------------------------------------------------

RECEPTOR_RULES: dict[str, dict[str, list[str]]] = {
    # --- Ionotropic glutamate receptors ---
    "AMPA": {
        "stimuli": ["NMDA", "D1", "5-HT2A", "alpha1A"],
        "inhibitory": ["GABA_A", "GABA_B", "CB1", "D2"],
    },
    "NMDA": {
        "stimuli": ["AMPA", "D1", "5-HT2A", "H2"],
        "inhibitory": [
            "GABA_A", "5-HT1A", "CB1", "sigma1", "GlyR alpha1"
        ],
    },
    "Kainate": {
        "stimuli": ["AMPA", "mGluR1"],
        "inhibitory": ["GABA_A"],
    },

    # --- Metabotropic glutamate receptors ---
    "mGluR1": {
        "stimuli": ["NMDA", "5-HT2A"],
        "inhibitory": ["GABA_B"],
    },
    "mGluR2": {
        # Glutamate activates mGluR2; AMPA/NMDA are not its ligands.
        # External ligand drive is not represented in this receptor-only graph.
        "stimuli": [],
        # Approximation: presynaptic mGluR2 activation can reduce glutamate
        # release and thereby reduce downstream AMPA/NMDA activity.
        "inhibitory": ["AMPA", "NMDA"],
    },

    # --- GABAergic receptors ---
    "GABA_A": {
        "stimuli": ["5-HT1A", "D2"],
        "inhibitory": ["AMPA", "NMDA", "H1", "NK1", "Mu (mi)"],
    },
    "GABA_B": {
        "stimuli": ["D2", "CB1"],
        "inhibitory": ["AMPA", "mGluR1", "H2"],
    },

    # --- Dopamine receptors ---
    "D1": {
        "stimuli": ["NMDA", "AMPA", "5-HT2A", "H1"],
        "inhibitory": ["GABA_A", "D2"],
    },
    "D2": {
        "stimuli": ["GABA_B", "5-HT1A", "H2"],
        "inhibitory": ["D1", "AMPA"],
    },

    # --- Serotonin receptors ---
    "5-HT1A": {
        "stimuli": ["GABA_A", "D2"],
        "inhibitory": ["NMDA", "5-HT2A"],
    },
    "5-HT2A": {
        "stimuli": ["D1", "NMDA", "mGluR1"],
        "inhibitory": ["GABA_A", "5-HT1A"],
    },
    "5-HT3": {
        # Circuit-level proxy: in some circuits, 5-HT3 activation excites
        # GABAergic interneurons. This is not a direct receptor interaction.
        "stimuli": ["GABA_A"],
        "inhibitory": [],
    },

    # --- Histamine receptors ---
    "H1": {
        "stimuli": ["D1", "5-HT2A"],
        "inhibitory": ["GABA_A", "MT1"],
    },
    "H2": {
        "stimuli": ["D2", "NMDA"],
        "inhibitory": ["GABA_B"],
    },
    "H3": {
        # H3 autoreceptor activation can suppress histamine release.
        # These downstream effects are circuit-level approximations.
        "stimuli": [],
        "inhibitory": ["H1", "H2"],
    },

    # --- Endocannabinoid receptors ---
    "CB1": {
        "stimuli": ["GABA_B"],
        # Approximation for reduced glutamate release at selected synapses.
        "inhibitory": ["NMDA", "AMPA"],
    },

    # --- Opioid receptors ---
    "Mu (mi)": {
        # Driven by external endogenous/exogenous opioid ligand input.
        "stimuli": [],
        "inhibitory": ["NMDA", "5-HT2A"],
    },

    # --- Other receptors ---
    "GlyR alpha1": {
        "stimuli": [],
        "inhibitory": ["NMDA"],
    },
    "alpha1A": {
        "stimuli": ["AMPA"],
        "inhibitory": [],
    },
    "alpha2A": {
        "stimuli": [],
        "inhibitory": ["D1", "5-HT2A"],
    },
    "sigma1": {
        "stimuli": [],
        "inhibitory": ["NMDA"],
    },
    "MT1": {
        "stimuli": [],
        "inhibitory": ["H1"],
    },
    "OX1": {
        # Circuit-level proxy for promoting histaminergic neuron activity.
        "stimuli": ["H1", "H2"],
        "inhibitory": [],
    },
    "NK1": {
        "stimuli": [],
        "inhibitory": ["GABA_A"],
    },
}


def validate_receptor_rules(
    rules: dict[str, dict[str, list[str]]],
    *,
    reject_conflicts: bool = True,
) -> None:
    """Raise ValueError for malformed rules, unknown names, or sign conflicts."""
    if not rules:
        raise ValueError("RECEPTOR_RULES must not be empty.")

    required_fields = {"stimuli", "inhibitory"}
    names = set(rules)

    for receptor, rule in rules.items():
        missing = required_fields - set(rule)
        extra = set(rule) - required_fields
        if missing:
            raise ValueError(f"{receptor}: missing fields {sorted(missing)}")
        if extra:
            raise ValueError(f"{receptor}: unexpected fields {sorted(extra)}")

        for field in required_fields:
            targets = rule[field]
            if not isinstance(targets, list):
                raise TypeError(f"{receptor}.{field} must be a list.")
            if len(targets) != len(set(targets)):
                raise ValueError(f"{receptor}.{field} contains duplicate entries.")
            unknown = set(targets) - names
            if unknown:
                raise ValueError(
                    f"{receptor}.{field} references undefined receptors: "
                    f"{sorted(unknown)}"
                )

        conflicts = set(rule["stimuli"]) & set(rule["inhibitory"])
        if conflicts and reject_conflicts:
            raise ValueError(
                f"{receptor}: receptors listed as both stimulatory and "
                f"inhibitory: {sorted(conflicts)}"
            )


validate_receptor_rules(RECEPTOR_RULES)

RECEPTOR_NAMES: list[str] = list(RECEPTOR_RULES)
N_RECEPTORS: int = len(RECEPTOR_NAMES)
_RECEPTOR_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(RECEPTOR_NAMES)
}


def _build_interaction_matrix(
    rules: dict[str, dict[str, list[str]]] = RECEPTOR_RULES,
) -> np.ndarray:
    """
    Build W with shape (target_receptors, source_receptors).

    W[target, source] = +1 if source stimulates target
                      = -1 if source inhibits target
                       =  0 if no connection is specified.
    """
    validate_receptor_rules(rules)
    names = list(rules)
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.float32)

    for target, rule in rules.items():
        target_i = index[target]
        for source in rule["stimuli"]:
            matrix[target_i, index[source]] = 1.0
        for source in rule["inhibitory"]:
            matrix[target_i, index[source]] = -1.0

    return matrix


INTERACTION_MATRIX: np.ndarray = _build_interaction_matrix()


# ---------------------------------------------------------------------------
# Elementwise Q-factor constraint
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="NeuronProject")
class ClipQ(tf.keras.constraints.Constraint):
    """Constrain each Q-factor independently to [min_value, max_value]."""

    def __init__(self, min_value: float = 0.1, max_value: float = 10.0):
        if min_value <= 0 or max_value < min_value:
            raise ValueError("Require 0 < min_value <= max_value.")
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(weights, self.min_value, self.max_value)

    def get_config(self) -> dict[str, float]:
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


# ---------------------------------------------------------------------------
# Keras layer
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="NeuronProject")
class NeuronLIKE(tf.keras.layers.Layer):
    """
    Receptor-interaction layer.

    Input/output shape: (batch, num_neurons, N_RECEPTORS).

    Q_in and Q_out are positive trainable per-neuron/per-receptor scale factors.
    The interaction matrix is fixed. A history ring buffer stores the mean input
    for each training call; it is non-trainable model state.
    """

    def __init__(
        self,
        num_neurons: int,
        history_size: int = 16,
        q_init_range: tuple[float, float] = (0.9, 1.1),
        threshold_init: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if not isinstance(num_neurons, int) or num_neurons <= 0:
            raise ValueError("num_neurons must be a positive integer.")
        if not isinstance(history_size, int) or history_size <= 0:
            raise ValueError("history_size must be a positive integer.")
        if len(q_init_range) != 2:
            raise ValueError("q_init_range must contain (low, high).")

        q_low, q_high = map(float, q_init_range)
        if q_low <= 0 or q_high < q_low:
            raise ValueError("Require 0 < q_init_range[0] <= q_init_range[1].")
        if q_low < 0.1 or q_high > 10.0:
            raise ValueError("q_init_range must lie within the Q constraint [0.1, 10].")

        self.num_neurons = num_neurons
        self.history_size = history_size
        self.q_init_range = (q_low, q_high)
        self.threshold_init_val = float(threshold_init)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_shape = tf.TensorShape(input_shape)

        if input_shape.rank != 3:
            raise ValueError(
                "NeuronLIKE expects rank-3 input "
                "(batch, num_neurons, N_RECEPTORS); "
                f"received shape {input_shape}."
            )

        neurons_dim = input_shape[1]
        receptors_dim = input_shape[2]

        if neurons_dim is not None and int(neurons_dim) != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} neurons, got {neurons_dim}."
            )
        if receptors_dim is not None and int(receptors_dim) != N_RECEPTORS:
            raise ValueError(
                f"Expected {N_RECEPTORS} receptor channels, got {receptors_dim}."
            )

        lo, hi = self.q_init_range
        q_constraint = ClipQ(0.1, 10.0)

        self.Q_in = self.add_weight(
            name="Q_in",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.RandomUniform(lo, hi),
            constraint=q_constraint,
            trainable=True,
        )
        self.Q_out = self.add_weight(
            name="Q_out",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.RandomUniform(lo, hi),
            constraint=ClipQ(0.1, 10.0),
            trainable=True,
        )
        self.threshold = self.add_weight(
            name="threshold",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.Constant(self.threshold_init_val),
            trainable=True,
        )
        self.W = self.add_weight(
            name="W",
            shape=(N_RECEPTORS, N_RECEPTORS),
            initializer=tf.keras.initializers.Constant(INTERACTION_MATRIX),
            trainable=False,
        )

        # add_weight makes this non-trainable state tracked by Keras.
        self._history_buffer = self.add_weight(
            name="history_buffer",
            shape=(self.num_neurons, self.history_size, N_RECEPTORS),
            initializer="zeros",
            trainable=False,
        )
        self._history_ptr = self.add_weight(
            name="history_ptr",
            shape=(),
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        inputs = tf.cast(tf.convert_to_tensor(inputs), self.compute_dtype)

        tf.debugging.assert_rank(
            inputs, 3,
            message="NeuronLIKE expects (batch, num_neurons, N_RECEPTORS).",
        )
        tf.debugging.assert_equal(
            tf.shape(inputs)[1],
            self.num_neurons,
            message="Input neuron dimension does not match num_neurons.",
        )
        tf.debugging.assert_equal(
            tf.shape(inputs)[2],
            N_RECEPTORS,
            message="Input receptor dimension does not match N_RECEPTORS.",
        )

        x_scaled = inputs * self.Q_in[tf.newaxis, :, :]

        # W[target, source]; sum over source to produce each target channel.
        mixed = tf.einsum("bnj,ij->bni", x_scaled, self.W)

        gate = tf.sigmoid(mixed - self.threshold[tf.newaxis, :, :])
        fired = mixed * gate
        output = fired * self.Q_out[tf.newaxis, :, :]

        # Update state only for an explicit Python True. This avoids attempting
        # a Python conditional on a symbolic boolean tensor under tf.function.
        if training is True:
            batch_mean = tf.reduce_mean(inputs, axis=0)  # (neurons, receptors)
            ptr = tf.cast(
                tf.math.floormod(self._history_ptr, self.history_size),
                tf.int32,
            )
            indices = tf.stack(
                [
                    tf.range(self.num_neurons, dtype=tf.int32),
                    tf.fill([self.num_neurons], ptr),
                ],
                axis=1,
            )
            self._history_buffer.assign(
                tf.tensor_scatter_nd_update(
                    self._history_buffer,
                    indices,
                    batch_mean,
                )
            )
            self._history_ptr.assign_add(1)

        return output

    def get_history_mean(self) -> tf.Tensor:
        """Return the mean of all slots in the history ring buffer."""
        return tf.reduce_mean(self._history_buffer, axis=1)

    def receptor_index(self, name: str) -> int:
        """Return the channel index for a receptor name."""
        try:
            return _RECEPTOR_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"Unknown receptor {name!r}.") from exc

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            num_neurons=self.num_neurons,
            history_size=self.history_size,
            q_init_range=list(self.q_init_range),
            threshold_init=self.threshold_init_val,
        )
        return config

    def save_weights_pickle(self, path: str | Path) -> None:
        """
        Save trainable parameters to a pickle file.

        Security: only load pickle files you created or otherwise trust.
        Use model.save() for full Keras model persistence.
        """
        data = {
            "format_version": 1,
            "num_neurons": self.num_neurons,
            "n_receptors": N_RECEPTORS,
            "receptor_names": RECEPTOR_NAMES,
            "Q_in": self.Q_in.numpy(),
            "Q_out": self.Q_out.numpy(),
            "threshold": self.threshold.numpy(),
        }
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights_pickle(self, path: str | Path) -> None:
        """Load compatible parameters from a trusted pickle file."""
        if not self.built:
            raise RuntimeError(
                "Build the layer first by calling it on an input tensor."
            )

        # WARNING: pickle can execute code. Never load an untrusted file.
        with open(path, "rb") as file:
            data = pickle.load(file)

        if data.get("format_version") != 1:
            raise ValueError("Unsupported or missing pickle format version.")
        if data.get("num_neurons") != self.num_neurons:
            raise ValueError("Saved num_neurons does not match this layer.")
        if data.get("n_receptors") != N_RECEPTORS:
            raise ValueError("Saved receptor count does not match this code.")
        if data.get("receptor_names") != RECEPTOR_NAMES:
            raise ValueError("Saved receptor order does not match this code.")

        expected_shapes = {
            "Q_in": tuple(self.Q_in.shape),
            "Q_out": tuple(self.Q_out.shape),
            "threshold": tuple(self.threshold.shape),
        }
        arrays: dict[str, np.ndarray] = {}
        for key, expected_shape in expected_shapes.items():
            if key not in data:
                raise ValueError(f"Missing {key!r} in saved weights.")
            array = np.asarray(data[key], dtype=np.float32)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{key} has shape {array.shape}; expected {expected_shape}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{key} contains NaN or infinite values.")
            arrays[key] = array

        self.Q_in.assign(arrays["Q_in"])
        self.Q_out.assign(arrays["Q_out"])
        self.threshold.assign(arrays["threshold"])


# ---------------------------------------------------------------------------
# Multi-layer network
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="NeuronProject")
class NeuronNetwork(tf.keras.Model):
    """Stack multiple NeuronLIKE layers with a standard Keras training loop."""

    def __init__(
        self,
        num_neurons: int,
        num_layers: int = 3,
        history_size: int = 16,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")

        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.history_size = history_size
        self.neuron_layers = [
            NeuronLIKE(
                num_neurons=num_neurons,
                history_size=history_size,
                name=f"neuron_layer_{i}",
            )
            for i in range(num_layers)
        ]

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        x = inputs
        for layer in self.neuron_layers:
            x = layer(x, training=training)
        return x

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            num_neurons=self.num_neurons,
            num_layers=self.num_layers,
            history_size=self.history_size,
        )
        return config


# ---------------------------------------------------------------------------
# Self-test / example
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """Run shape, gradient, history, and save/load smoke tests."""
    batch_size = 4
    num_neurons = 3

    sample_input = tf.random.uniform(
        (batch_size, num_neurons, N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )
    target = tf.random.uniform(
        (batch_size, num_neurons, N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    # Validate receptor rules and matrix dimensions.
    validate_receptor_rules(RECEPTOR_RULES)
    assert INTERACTION_MATRIX.shape == (N_RECEPTORS, N_RECEPTORS)
    assert np.isfinite(INTERACTION_MATRIX).all()

    layer = NeuronLIKE(num_neurons=num_neurons, history_size=16)
    output = layer(sample_input, training=True)
    assert tuple(output.shape) == (batch_size, num_neurons, N_RECEPTORS)
    assert tuple(layer.get_history_mean().shape) == (num_neurons, N_RECEPTORS)

    # Test gradient flow through the multi-layer model.
    model = NeuronNetwork(
        num_neurons=num_neurons,
        num_layers=2,
        history_size=16,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    with tf.GradientTape() as tape:
        prediction = model(sample_input, training=True)
        loss = tf.reduce_mean(tf.square(prediction - target))

    gradients = tape.gradient(loss, model.trainable_variables)
    if not gradients:
        raise RuntimeError("No gradients were produced.")
    missing_gradients = [
        variable.name
        for variable, gradient in zip(model.trainable_variables, gradients)
        if gradient is None
    ]
    if missing_gradients:
        raise RuntimeError(f"Missing gradients for: {missing_gradients}")

    # Verify Q constraints are finite and in range.
    for q in (layer.Q_in, layer.Q_out):
        if not bool(tf.reduce_all(tf.math.is_finite(q)).numpy()):
            raise RuntimeError("Non-finite Q-factor found.")
        if bool(tf.reduce_any(q < 0.1).numpy()) or bool(
            tf.reduce_any(q > 10.0).numpy()
        ):
            raise RuntimeError("Q-factor outside the permitted range.")

    # Test trusted pickle round-trip.
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "neuron_weights.pkl"
        layer.save_weights_pickle(path)
        layer.load_weights_pickle(path)

    print("Self-test passed.")
    print(f"Receptors: {N_RECEPTORS}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Loss: {float(loss.numpy()):.6f}")
    print(f"Trainable variables: {len(model.trainable_variables)}")


if __name__ == "__main__":
    run_self_test()


# # ---------------------------------------------------------------------------
# # Receptor interaction rules
# # Each receptor lists which other receptors stimulate or inhibit it.
# # ---------------------------------------------------------------------------
# RECEPTOR_RULES: dict[str, dict] = {
#     # --- Ionotropic glutamate receptors ---
#     "AMPA": {
#         "stimuli": ["NMDA", "D1", "5-HT2A", "alpha1A"],
#         "inhibitory": ["GABA_A", "GABA_B", "CB1", "D2"],
#     },
#     "NMDA": {
#         "stimuli": ["AMPA", "D1", "5-HT2A", "H2"],
#         "inhibitory": [
#             "GABA_A", "5-HT1A", "CB1", "sigma1", "GlyR alpha1"
#         ],
#     },
#     "Kainate": {
#         "stimuli": ["AMPA", "mGluR1"],
#         "inhibitory": ["GABA_A"],
#     },

#     # --- Metabotropic glutamate receptors ---
#     "mGluR1": {
#         "stimuli": ["NMDA", "5-HT2A"],
#         "inhibitory": ["GABA_B"],
#     },
#     "mGluR2": {
#         # AMPA/NMDA activity is only an indirect proxy for glutamate tone.
#         "stimuli": ["AMPA", "NMDA"],
#         # Presynaptic mGluR2 activation can reduce glutamate release.
#         # AMPA/NMDA represent downstream effects, not direct targets.
#         "inhibitory": ["AMPA", "NMDA"],
#     },

#     # --- GABAergic receptors ---
#     "GABA_A": {
#         "stimuli": ["5-HT1A", "D2"],
#         "inhibitory": ["AMPA", "NMDA", "H1", "NK1", "Mu (mi)"],
#     },
#     "GABA_B": {
#         "stimuli": ["D2", "CB1"],
#         "inhibitory": ["AMPA", "mGluR1", "H2"],
#     },

#     # --- Dopamine receptors ---
#     "D1": {
#         "stimuli": ["NMDA", "AMPA", "5-HT2A", "H1"],
#         "inhibitory": ["GABA_A", "D2"],
#     },
#     "D2": {
#         "stimuli": ["GABA_B", "5-HT1A", "H2"],
#         "inhibitory": ["D1", "AMPA"],
#     },

#     # --- Serotonin receptors ---
#     "5-HT1A": {
#         "stimuli": ["GABA_A", "D2"],
#         "inhibitory": ["NMDA", "5-HT2A"],
#     },
#     "5-HT2A": {
#         "stimuli": ["D1", "NMDA", "mGluR1"],
#         "inhibitory": ["GABA_A", "5-HT1A"],
#     },
#     "5-HT3": {
#         # Circuit-level proxy: 5-HT3 activation can excite
#         # GABAergic interneurons in particular circuits.
#         "stimuli": ["GABA_A"],
#         "inhibitory": [],
#     },

#     # --- Histamine receptors ---
#     "H1": {
#         "stimuli": ["D1", "5-HT2A"],
#         "inhibitory": ["GABA_A", "MT1"],
#     },
#     "H2": {
#         "stimuli": ["D2", "NMDA"],
#         "inhibitory": ["GABA_B"],
#     },
#     "H3": {
#         # Histamine autoreceptor: suppresses histamine release.
#         # The effects on H1/H2 are indirect and depend on the circuit.
#         "stimuli": [],
#         "inhibitory": ["H1", "H2"],
#     },

#     # --- Endocannabinoid receptors ---
#     "CB1": {
#         "stimuli": ["GABA_B"],
#         # Circuit-level proxy for reduced glutamate release.
#         "inhibitory": ["NMDA", "AMPA"],
#     },

#     # --- Opioid receptors ---
#     "Mu (mi)": {
#         # Externally driven by endogenous or exogenous opioid ligands.
#         "stimuli": [],
#         "inhibitory": ["NMDA", "5-HT2A"],
#     },

#     # --- Other receptors ---
#     "GlyR alpha1": {
#         "stimuli": [],
#         "inhibitory": ["NMDA"],
#     },
#     "alpha1A": {
#         "stimuli": ["AMPA"],
#         "inhibitory": [],
#     },
#     "alpha2A": {
#         "stimuli": [],
#         "inhibitory": ["D1", "5-HT2A"],
#     },
#     "sigma1": {
#         "stimuli": [],
#         "inhibitory": ["NMDA"],
#     },
#     "MT1": {
#         "stimuli": [],
#         "inhibitory": ["H1"],
#     },
#     "OX1": {
#         # Circuit-level proxy for promoting histaminergic activity.
#         "stimuli": ["H1", "H2"],
#         "inhibitory": [],
#     },
#     "NK1": {
#         "stimuli": [],
#         "inhibitory": ["GABA_A"],
#     },
# }

# RECEPTOR_NAMES: list[str] = list(RECEPTOR_RULES.keys())
# N_RECEPTORS: int = len(RECEPTOR_NAMES)
# _RECEPTOR_INDEX: dict[str, int] = {name: i for i, name in enumerate(RECEPTOR_NAMES)}


# def _build_interaction_matrix() -> np.ndarray:
#     """
#     Build a fixed (N_RECEPTORS × N_RECEPTORS) interaction matrix W where:
#         W[i, j] = +1  if receptor j stimulates receptor i
#         W[i, j] = -1  if receptor j inhibits  receptor i
#         W[i, j] =  0  otherwise

#     This lets the full receptor response be computed as a single matmul:
#         response = W @ inputs          (before Q scaling)
#     """
#     W = np.zeros((N_RECEPTORS, N_RECEPTORS), dtype=np.float32)
#     for receptor, rules in RECEPTOR_RULES.items():
#         i = _RECEPTOR_INDEX[receptor]
#         for src in rules.get("stimuli", []):
#             if src in _RECEPTOR_INDEX:
#                 W[i, _RECEPTOR_INDEX[src]] = 1.0
#         for src in rules.get("inhibitory", []):
#             if src in _RECEPTOR_INDEX:
#                 W[i, _RECEPTOR_INDEX[src]] = -1.0
#     return W


# # Precomputed once at import time; never trained.
# INTERACTION_MATRIX: np.ndarray = _build_interaction_matrix()


# # ---------------------------------------------------------------------------
# # Keras Layer
# # ---------------------------------------------------------------------------

# class NeuronLIKE(tf.keras.layers.Layer):
#     """
#     A biologically-inspired dense layer whose units (neurons) each maintain
#     a pair of trainable Q-factors per receptor type.

#     Input shape:  (batch, num_neurons, N_RECEPTORS)   — one activation per
#                   receptor channel per neuron.
#     Output shape: (batch, num_neurons, N_RECEPTORS)

#     Internal variables
#     ------------------
#     Q_in  : (num_neurons, N_RECEPTORS)  — scales incoming receptor signals
#     Q_out : (num_neurons, N_RECEPTORS)  — scales outgoing receptor signals
#     W     : fixed (N_RECEPTORS, N_RECEPTORS) interaction matrix (non-trainable)

#     Forward pass
#     ------------
#     1. Scale inputs:      x_scaled = x * Q_in            [per neuron, per receptor]
#     2. Mix receptors:     mixed    = x_scaled @ W.T       [receptor cross-talk]
#     3. Apply threshold:   fired    = mixed * sigmoid(mixed - threshold)
#     4. Scale outputs:     out      = fired * Q_out
#     """

#     def __init__(
#         self,
#         num_neurons: int,
#         history_size: int = 16,
#         q_init_range: tuple[float, float] = (0.9, 1.1),
#         threshold_init: float = 0.0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.num_neurons = num_neurons
#         self.history_size = history_size
#         self.q_init_range = q_init_range
#         self.threshold_init_val = threshold_init

#     # ------------------------------------------------------------------
#     # Layer building (called once on first call with real input shape)
#     # ------------------------------------------------------------------

#     def build(self, input_shape):
#         lo, hi = self.q_init_range

#         # Trainable quality factors — initialised near 1 so the layer starts
#         # close to a standard linear transform.
#         self.Q_in = self.add_weight(
#             name="Q_in",
#             shape=(self.num_neurons, N_RECEPTORS),
#             initializer=tf.keras.initializers.RandomUniform(lo, hi),
#             constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=10.0),
#             trainable=True,
#         )
#         self.Q_out = self.add_weight(
#             name="Q_out",
#             shape=(self.num_neurons, N_RECEPTORS),
#             initializer=tf.keras.initializers.RandomUniform(lo, hi),
#             constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=10.0),
#             trainable=True,
#         )

#         # Per-neuron, per-receptor learned firing threshold
#         self.threshold = self.add_weight(
#             name="threshold",
#             shape=(self.num_neurons, N_RECEPTORS),
#             initializer=tf.keras.initializers.Constant(self.threshold_init_val),
#             trainable=True,
#         )

#         # Fixed interaction matrix — not trained
#         self.W = self.add_weight(
#             name="W",
#             shape=(N_RECEPTORS, N_RECEPTORS),
#             initializer=tf.keras.initializers.Constant(INTERACTION_MATRIX),
#             trainable=False,
#         )

#         # Ring buffer for input history (not a tf.Variable so it won't be
#         # serialised as a weight; reset on each build/call explicitly).
#         # Shape: (num_neurons, history_size, N_RECEPTORS)
#         self._history_buffer = tf.Variable(
#             tf.zeros((self.num_neurons, self.history_size, N_RECEPTORS)),
#             trainable=False,
#             name="history_buffer",
#         )
#         self._history_ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name="history_ptr")

#         super().build(input_shape)

#     # ------------------------------------------------------------------
#     # Forward pass
#     # ------------------------------------------------------------------

#     def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
#         """
#         Parameters
#         ----------
#         inputs : tf.Tensor, shape (batch, num_neurons, N_RECEPTORS)
#             Raw receptor activation values.
#         training : bool
#             When True, history buffer is updated (useful for analysis).

#         Returns
#         -------
#         tf.Tensor, shape (batch, num_neurons, N_RECEPTORS)
#         """
#         # 1. Scale incoming signals
#         x_scaled = inputs * self.Q_in[tf.newaxis, :, :]   # (batch, neurons, R)

#         # 2. Receptor cross-talk via interaction matrix
#         #    einsum: for each sample b and neuron n, mix[b,n,i] = sum_j x_scaled[b,n,j]*W[j,i]
#         mixed = tf.einsum("bnj,ij->bni", x_scaled, self.W)  # (batch, neurons, R)

#         # 3. Soft threshold — smooth surrogate for "fire if above threshold"
#         #    Using sigmoid so the operation is fully differentiable.
#         gate = tf.sigmoid(mixed - self.threshold[tf.newaxis, :, :])
#         fired = mixed * gate                                 # (batch, neurons, R)

#         # 4. Scale output
#         output = fired * self.Q_out[tf.newaxis, :, :]       # (batch, neurons, R)

#         # 5. Optionally update history ring buffer (batch mean stored)
#         if training:
#             batch_mean = tf.reduce_mean(inputs, axis=0)     # (neurons, R)
#             ptr = self._history_ptr % self.history_size
#             indices = tf.stack(
#                 [tf.range(self.num_neurons),
#                  tf.fill([self.num_neurons], ptr)], axis=1
#             )
#             self._history_buffer.scatter_nd_update(indices, batch_mean)
#             self._history_ptr.assign_add(1)

#         return output

#     # ------------------------------------------------------------------
#     # Convenience helpers
#     # ------------------------------------------------------------------

#     def get_history_mean(self) -> tf.Tensor:
#         """Return the running mean of inputs over the history buffer."""
#         return tf.reduce_mean(self._history_buffer, axis=1)  # (neurons, R)

#     def receptor_index(self, name: str) -> int:
#         """Return the column index for a given receptor name."""
#         return _RECEPTOR_INDEX[name]

#     # ------------------------------------------------------------------
#     # Serialisation
#     # ------------------------------------------------------------------

#     def get_config(self) -> dict:
#         cfg = super().get_config()
#         cfg.update(
#             num_neurons=self.num_neurons,
#             history_size=self.history_size,
#             q_init_range=self.q_init_range,
#             threshold_init=self.threshold_init_val,
#         )
#         return cfg

#     def save_weights_pickle(self, path: str) -> None:
#         """Lightweight save: pickle just the numpy weight arrays."""
#         data = {
#             "Q_in": self.Q_in.numpy(),
#             "Q_out": self.Q_out.numpy(),
#             "threshold": self.threshold.numpy(),
#         }
#         with open(path, "wb") as f:
#             pickle.dump(data, f)

#     def load_weights_pickle(self, path: str) -> None:
#         """Load weights previously saved with save_weights_pickle."""
#         with open(path, "rb") as f:
#             data = pickle.load(f)
#         self.Q_in.assign(data["Q_in"])
#         self.Q_out.assign(data["Q_out"])
#         self.threshold.assign(data["threshold"])


# # ---------------------------------------------------------------------------
# # Multi-layer network helper
# # ---------------------------------------------------------------------------

# class NeuronNetwork(tf.keras.Model):
#     """
#     Stack several NeuronLIKEs into a Keras Model.

#     Input:  (batch, num_neurons, N_RECEPTORS)
#     Output: (batch, num_neurons, N_RECEPTORS)

#     Training uses standard Keras model.compile / model.fit.
#     Loss is MSE between output receptor activations and the target.
#     """

#     def __init__(self, num_neurons: int, num_layers: int = 3, history_size: int = 16, **kwargs):
#         super().__init__(**kwargs)
#         self.neuron_layers = [
#             NeuronLIKE(num_neurons, history_size=history_size, name=f"neuron_layer_{i}")
#             for i in range(num_layers)
#         ]

#     def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
#         x = inputs
#         for layer in self.neuron_layers:
#             x = layer(x, training=training)
#         return x


# # ---------------------------------------------------------------------------
# # Sample usage
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     BATCH = 4
#     NUM_NEURONS = 3

#     # Random receptor activations — shape (batch, neurons, receptors)
#     sample_input = tf.random.uniform((BATCH, NUM_NEURONS, N_RECEPTORS), minval=0.0, maxval=1.0)

#     # --- Single layer ---
#     layer = NeuronLIKE(num_neurons=NUM_NEURONS, history_size=16)
#     out = layer(sample_input, training=True)
#     print("NeuronLIKE output shape:", out.shape)          # (4, 3, 24)
#     print("Q_in  shape:", layer.Q_in.shape)                # (3, 24)
#     print("History mean shape:", layer.get_history_mean().shape)

#     # --- Multi-layer network with Keras training loop ---
#     model = NeuronNetwork(num_neurons=NUM_NEURONS, num_layers=2)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         loss=tf.keras.losses.MeanSquaredError(),
#     )

#     # Dummy supervised target: same shape as output
#     target = tf.random.uniform((BATCH, NUM_NEURONS, N_RECEPTORS))

#     # Single training step to verify gradients flow
#     with tf.GradientTape() as tape:
#         pred = model(sample_input, training=True)
#         loss = tf.reduce_mean(tf.square(pred - target))

#     grads = tape.gradient(loss, model.trainable_variables)
#     print(f"\nTraining step — loss: {loss.numpy():.6f}")
#     for var, grad in zip(model.trainable_variables, grads):
#         print(f"  {var.name:40s}  grad norm: {tf.norm(grad).numpy():.4f}")

#     # Save / load weights
#     layer.save_weights_pickle("/tmp/neuron_weights.pkl")
#     layer.load_weights_pickle("/tmp/neuron_weights.pkl")
#     print("\nWeights saved and reloaded successfully.")

#     # Receptor index lookup
#     print(f"\nAMPA receptor index: {layer.receptor_index('AMPA')}")
#     print(f"NMDA receptor index: {layer.receptor_index('NMDA')}")
