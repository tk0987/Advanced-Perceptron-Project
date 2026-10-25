# """
# Hybrid CNN + Dense + receptor-mixing neuron network.

# Architecture:
#     Input image
#         ├── CNN feature extractor ──> selected excitatory receptors
#         └── Dense feature extractor -> selected inhibitory/modulatory receptors
#                                            |
#                                       Receptor mixer
#                                            |
#                                       NeuronLIKE (12)
#                                            |
#                                       NeuronLIKE (12)
#                                            |
#                                          Flatten
#                                            |
#                                        Dense (32)
#                                            |
#                                      Dense (10 classes)

# Requirements:
#     numpy
#     tensorflow
#     neuron_project_fixed.py
# """

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model

# import neuron_like as neuron


# # ============================================================
# # Configuration
# # ============================================================

# INPUT_SHAPE = (28, 28, 1)
# NUM_NEURONS = 64
# NUM_CLASSES = 10
# BATCH_SIZE = 64
# EPOCHS = 40

# N_RECEPTORS = neuron.N_RECEPTORS
# RECEPTOR_NAMES = neuron.RECEPTOR_NAMES


# # ============================================================
# # Receptor routing layer
# # ============================================================

# @tf.keras.utils.register_keras_serializable(
#     package="NeuronProject"
# )
# class ReceptorMask(layers.Layer):
#     """
#     Route activations to selected receptor channels.

#     Input:
#         (batch, neurons, receptors)

#     Output:
#         Same shape, with unselected receptor channels set to zero.
#     """

#     def __init__(self, mask_values, **kwargs):
#         super().__init__(**kwargs)

#         if len(mask_values) != N_RECEPTORS:
#             raise ValueError(
#                 f"Expected {N_RECEPTORS} mask values, "
#                 f"got {len(mask_values)}."
#             )

#         self.mask_values = [float(x) for x in mask_values]

#     def call(self, inputs):
#         mask = tf.constant(
#             self.mask_values,
#             dtype=inputs.dtype,
#         )
#         return inputs * mask

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "mask_values": self.mask_values,
#         })
#         return config


# def make_receptor_mask(receptor_names):
#     """
#     Create a binary mask selecting named receptor channels.
#     """
#     unknown = set(receptor_names) - set(RECEPTOR_NAMES)

#     if unknown:
#         raise ValueError(
#             f"Unknown receptors: {sorted(unknown)}"
#         )

#     selected = set(receptor_names)

#     return [
#         1.0 if name in selected else 0.0
#         for name in RECEPTOR_NAMES
#     ]


# # ============================================================
# # Network construction
# # ============================================================

# def build_hybrid_network():
#     image = layers.Input(
#         shape=INPUT_SHAPE,
#         name="image",
#     )

#     # --------------------------------------------------------
#     # Branch A: convolutional feature extractor
#     # --------------------------------------------------------

#     conv = layers.Conv2D(
#         16,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#         name="conv_1",
#     )(image)

#     conv = layers.MaxPooling2D(
#         pool_size=2,
#         name="conv_pool",
#     )(conv)

#     conv = layers.Conv2D(
#         24,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#         name="conv_2",
#     )(conv)

#     conv = layers.GlobalAveragePooling2D(
#         name="conv_global_pool",
#     )(conv)

#     # --------------------------------------------------------
#     # Branch B: conventional dense feature extractor
#     # --------------------------------------------------------

#     dense = layers.Flatten(
#         name="input_flatten",
#     )(image)

#     dense = layers.Dense(
#         32,
#         activation="relu",
#         name="feature_dense",
#     )(dense)

#     # --------------------------------------------------------
#     # Project features into receptor signals.
#     #
#     # Each branch produces a tensor of shape:
#     #     (batch, NUM_NEURONS, N_RECEPTORS)
#     #
#     # The masks determine which receptors receive each branch.
#     # --------------------------------------------------------

#     conv_signals = layers.Dense(
#         NUM_NEURONS * N_RECEPTORS,
#         activation="sigmoid",
#         name="conv_receptor_projection",
#     )(conv)

#     conv_signals = layers.Reshape(
#         (NUM_NEURONS, N_RECEPTORS),
#         name="conv_receptor_reshape",
#     )(conv_signals)

#     dense_signals = layers.Dense(
#         NUM_NEURONS * N_RECEPTORS,
#         activation="sigmoid",
#         name="dense_receptor_projection",
#     )(dense)

#     dense_signals = layers.Reshape(
#         (NUM_NEURONS, N_RECEPTORS),
#         name="dense_receptor_reshape",
#     )(dense_signals)

#     # Example receptor assignments.
#     #
#     # CNN features -> glutamate receptor channels.
#     # Dense features -> inhibitory/modulatory channels.
#     #
#     # These assignments are illustrative, not a claim that
#     # these pathways reproduce real biological circuitry.

#     conv_mask = make_receptor_mask([
#         "AMPA",
#         "NMDA",
#         "Kainate",
#     ])

#     dense_mask = make_receptor_mask([
#         "GABA_A",
#         "GABA_B",
#         "D1",
#         "D2",
#         "GlyR alpha1",
#     ])

#     conv_signals = ReceptorMask(
#         conv_mask,
#         name="conv_receptor_routing",
#     )(conv_signals)

#     dense_signals = ReceptorMask(
#         dense_mask,
#         name="dense_receptor_routing",
#     )(dense_signals)

#     # Combine both branches into a single receptor tensor.
#     receptor_input = layers.Add(
#         name="receptor_signal_mixer",
#     )([
#         conv_signals,
#         dense_signals,
#     ])

#     # Shape:
#     # (batch, 12, N_RECEPTORS)

#     # --------------------------------------------------------
#     # Two neuron layers from your existing implementation
#     # --------------------------------------------------------

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=32,
#         name="neuron_layer_1",
#     )(receptor_input)

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=32,
#         name="neuron_layer_2",
#     )(x)
#     x = neuron.NeuronLIKE(
#             num_neurons=NUM_NEURONS,
#             history_size=32,
#             name="neuron_layer_3",
#         )(x)
#     x = neuron.NeuronLIKE(
#                 num_neurons=NUM_NEURONS,
#                 history_size=32,
#                 name="neuron_layer_4",
#             )(x)
#     # --------------------------------------------------------
#     # Conventional output head
#     # --------------------------------------------------------

#     x = layers.Flatten(
#         name="neuron_output_flatten",
#     )(x)

#     x = layers.Dense(
#         32,
#         activation="relu",
#         name="output_hidden",
#     )(x)

#     output = layers.Dense(
#         NUM_CLASSES,
#         activation="softmax",
#         name="classification_output",
#     )(x)

#     model = Model(
#         inputs=image,
#         outputs=output,
#         name="hybrid_neuron_network",
#     )

#     return model


# # ============================================================
# # Run an example
# # ============================================================

# def main():
#     print("TensorFlow:", tf.__version__)
#     print("Receptors:", N_RECEPTORS)
#     print("Neuron count per layer:", NUM_NEURONS)

#     model = build_hybrid_network()

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=1e-4,
#         ),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()

#     # Load MNIST images (28x28 grayscale, 10 classes).
#     (x_train, y_train), (x_test, y_test) = (
#         tf.keras.datasets.mnist.load_data()
#     )

#     # Normalize pixels to [0, 1].
#     x_train = (
#         x_train.astype(np.float32) / 255.0
#     )[..., np.newaxis]

#     x_test = (
#         x_test.astype(np.float32) / 255.0
#     )[..., np.newaxis]

#     # Small subset for a quick demonstration.
#     # Increase or remove these slices for a full run.
#     x_train = x_train[:10000]
#     y_train = y_train[:10000]

#     x_test = x_test[:5000]
#     y_test = y_test[:5000]

#     print("\nStarting training...")

#     history = model.fit(
#         x_train,
#         y_train,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         validation_split=0.1,
#         verbose=1,
#     )
#     for layer in model.layers:
#         if isinstance(layer, neuron.NeuronLIKE):
#             print(f"\n{layer.name}")

#             for name in ("Q_in", "Q_out", "threshold"):
#                 value = getattr(layer, name).numpy()

#                 print(
#                     f"  {name}: "
#                     f"min={value.min():.4f}, "
#                     f"mean={value.mean():.4f}, "
#                     f"max={value.max():.4f}, "
#                     f"std={value.std():.4f}"
#                 )


#     for layer in model.layers:
#         if not isinstance(layer, neuron.NeuronLIKE):
#             continue

#         print(f"\n{'=' * 60}")
#         print(layer.name)
#         print("=" * 60)

#         q_in = layer.Q_in.numpy()
#         q_out = layer.Q_out.numpy()
#         thresholds = layer.threshold.numpy()

#         for receptor_index, receptor_name in enumerate(
#             neuron.RECEPTOR_NAMES
#         ):
#             print(
#                 f"{receptor_name:15s} | "
#                 f"Q_in mean={q_in[:, receptor_index].mean():.4f} | "
#                 f"Q_out mean={q_out[:, receptor_index].mean():.4f} | "
#                 f"threshold mean="
#                 f"{thresholds[:, receptor_index].mean():.4f}"
#             )
#     print("\nEvaluating...")
#     loss, accuracy = model.evaluate(
#         x_test,
#         y_test,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#     )

#     print(f"\nTest loss:     {loss:.4f}")
#     print(f"Test accuracy: {accuracy:.4f}")

#     # Inspect predictions for a few images.
#     predictions = model.predict(
#         x_test[:5],
#         verbose=0,
#     )

#     print("\nSample predictions:")

#     for i, probabilities in enumerate(predictions):
#         predicted_class = int(np.argmax(probabilities))

#         print(
#             f"Image {i}: "
#             f"predicted={predicted_class}, "
#             f"actual={int(y_test[i])}, "
#             f"confidence={float(np.max(probabilities)):.3f}"
#         )

#     # Save the complete Keras model.
#     model.save("hybrid_neuron_net.keras")

#     print("\nSaved: hybrid_neuron_net.keras")


# if __name__ == "__main__":
#     main()
"""
Hybrid CNN + Dense + receptor-mixing neuron network.

Architecture:
    Input image
        ├── CNN feature extractor
        │       └── CNN receptor channels
        │
        └── Dense feature extractor
                └── Dense receptor channels

                       ↓
                Receptor signal mixer
                       ↓
                  NeuronLIKE (64)
                       ↓
                  NeuronLIKE (64)
                       ↓
                  NeuronLIKE (64)
                       ↓
                  NeuronLIKE (64)
                       ↓
                     Flatten
                       ↓
                   Dense (32)
                       ↓
                 Dense (10 classes)

Important:
    CNN and Dense branches use disjoint receptor channels
    at the input to the first NeuronLIKE layer.

    After the first layer, all 24 receptor channels are
    available to every subsequent NeuronLIKE layer.

Requirements:
    numpy
    tensorflow
    neuron_like.py
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model

import neuron_like as neuron


# ============================================================
# Configuration
# ============================================================

INPUT_SHAPE = (28, 28, 1)

NUM_NEURONS = 64
NUM_CLASSES = 10

BATCH_SIZE = 64
EPOCHS = 40

LEARNING_RATE = 5e-4

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 5000

HISTORY_SIZE = 32

N_RECEPTORS = neuron.N_RECEPTORS
RECEPTOR_NAMES = list(neuron.RECEPTOR_NAMES)


# ============================================================
# Receptor assignments
# ============================================================

# CNN branch:
# Signals from the convolutional feature extractor are
# assigned to these receptor channels.

CNN_RECEPTORS = [
    "AMPA",
    "NMDA",
    "Kainate",
    "mGluR1",
    "D1",
    "5-HT2A",
]


# Dense branch:
# Signals from the conventional dense feature extractor
# are assigned to a separate set of receptor channels.

DENSE_RECEPTORS = [
    "GABA_A",
    "GABA_B",
    "mGluR2",
    "5-HT1A",
    "CB1",
    "GlyR alpha1",
    "alpha2A",
    "sigma1",
    "MT1",
]


# ============================================================
# Validate receptor assignments
# ============================================================

def validate_receptor_assignments():
    """
    Check that all receptor names exist and that the CNN and
    Dense branches have no receptor channels in common.
    """

    available = set(RECEPTOR_NAMES)

    cnn_set = set(CNN_RECEPTORS)
    dense_set = set(DENSE_RECEPTORS)

    unknown_cnn = cnn_set - available
    unknown_dense = dense_set - available

    if unknown_cnn:
        raise ValueError(
            f"Unknown CNN receptors: {sorted(unknown_cnn)}"
        )

    if unknown_dense:
        raise ValueError(
            f"Unknown Dense receptors: {sorted(unknown_dense)}"
        )

    overlap = cnn_set & dense_set

    if overlap:
        raise ValueError(
            "CNN and Dense branches must use different "
            f"receptor channels. Overlap: {sorted(overlap)}"
        )

    if not cnn_set:
        raise ValueError("CNN_RECEPTORS cannot be empty.")

    if not dense_set:
        raise ValueError("DENSE_RECEPTORS cannot be empty.")

    unassigned = available - cnn_set - dense_set

    print("\nReceptor routing configuration")
    print("=" * 60)

    print(f"CNN receptors ({len(cnn_set)}):")
    for name in CNN_RECEPTORS:
        print(f"  + {name}")

    print(f"\nDense receptors ({len(dense_set)}):")
    for name in DENSE_RECEPTORS:
        print(f"  + {name}")

    print(f"\nUnassigned input channels ({len(unassigned)}):")
    for name in RECEPTOR_NAMES:
        if name in unassigned:
            print(f"  - {name}")

    print("\nThe two branches use disjoint receptor channels.")
    print("All neuron layers retain all receptor channels.")


# ============================================================
# Receptor routing layer
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="NeuronProject"
)
class ReceptorMask(layers.Layer):
    """
    Mask selected receptor channels.

    Input:
        (batch, neurons, receptors)

    Output:
        (batch, neurons, receptors)

    Channels whose mask value is zero are set to zero.
    """

    def __init__(self, mask_values, **kwargs):
        super().__init__(**kwargs)

        if len(mask_values) != N_RECEPTORS:
            raise ValueError(
                f"Expected {N_RECEPTORS} mask values, "
                f"got {len(mask_values)}."
            )

        self.mask_values = [float(x) for x in mask_values]

    def call(self, inputs):
        mask = tf.constant(
            self.mask_values,
            dtype=inputs.dtype,
        )

        return inputs * mask

    def get_config(self):
        config = super().get_config()

        config.update({
            "mask_values": self.mask_values,
        })

        return config


def make_receptor_mask(receptor_names):
    """
    Create a binary mask in RECEPTOR_NAMES order.
    """

    unknown = set(receptor_names) - set(RECEPTOR_NAMES)

    if unknown:
        raise ValueError(
            f"Unknown receptors: {sorted(unknown)}"
        )

    selected = set(receptor_names)

    return [
        1.0 if name in selected else 0.0
        for name in RECEPTOR_NAMES
    ]


# ============================================================
# Network construction
# ============================================================

# def build_hybrid_network():

#     # Validate the user-configured assignments before building.
#     validate_receptor_assignments()

#     image = layers.Input(
#         shape=INPUT_SHAPE,
#         name="image",
#     )

#     # --------------------------------------------------------
#     # Branch A: CNN feature extractor
#     # --------------------------------------------------------

#     conv = layers.Conv2D(
#         filters=16,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#         name="conv_1",
#     )(image)

#     conv = layers.MaxPooling2D(
#         pool_size=2,
#         name="conv_pool",
#     )(conv)

#     conv = layers.Conv2D(
#         filters=24,
#         kernel_size=3,
#         padding="same",
#         activation="relu",
#         name="conv_2",
#     )(conv)

#     conv = layers.GlobalAveragePooling2D(
#         name="conv_global_pool",
#     )(conv)

#     # --------------------------------------------------------
#     # Branch B: Dense feature extractor
#     # --------------------------------------------------------

#     dense = layers.Flatten(
#         name="input_flatten",
#     )(image)

#     dense = layers.Dense(
#         units=32,
#         activation="relu",
#         name="feature_dense",
#     )(dense)

#     # --------------------------------------------------------
#     # Project CNN features into receptor signals.
#     #
#     # Shape:
#     # (batch, NUM_NEURONS, N_RECEPTORS)
#     #
#     # The projection has a separate learned output for every
#     # neuron and receptor channel.
#     # --------------------------------------------------------

#     conv_signals = layers.Dense(
#         units=NUM_NEURONS * N_RECEPTORS,
#         activation="sigmoid",
#         name="conv_receptor_projection",
#     )(conv)

#     conv_signals = layers.Reshape(
#         (NUM_NEURONS, N_RECEPTORS),
#         name="conv_receptor_reshape",
#     )(conv_signals)

#     # --------------------------------------------------------
#     # Project Dense features into receptor signals.
#     #
#     # This is a separate trainable projection.
#     # --------------------------------------------------------

#     dense_signals = layers.Dense(
#         units=NUM_NEURONS * N_RECEPTORS,
#         activation="sigmoid",
#         name="dense_receptor_projection",
#     )(dense)

#     dense_signals = layers.Reshape(
#         (NUM_NEURONS, N_RECEPTORS),
#         name="dense_receptor_reshape",
#     )(dense_signals)

#     # --------------------------------------------------------
#     # Route CNN and Dense branches to different receptors.
#     # --------------------------------------------------------

#     conv_mask = make_receptor_mask(CNN_RECEPTORS)
#     dense_mask = make_receptor_mask(DENSE_RECEPTORS)

#     conv_signals = ReceptorMask(
#         conv_mask,
#         name="conv_receptor_routing",
#     )(conv_signals)

#     dense_signals = ReceptorMask(
#         dense_mask,
#         name="dense_receptor_routing",
#     )(dense_signals)

#     # --------------------------------------------------------
#     # Combine the branches.
#     #
#     # Because the masks are disjoint, the CNN branch supplies
#     # its selected channels and the Dense branch supplies its
#     # own selected channels.
#     #
#     # Shape:
#     # (batch, NUM_NEURONS, N_RECEPTORS)
#     # --------------------------------------------------------

#     receptor_input = layers.Add(
#         name="receptor_signal_mixer",
#     )([
#         conv_signals,
#         dense_signals,
#     ])

#     # --------------------------------------------------------
#     # Neuron layer 1
#     #
#     # Receives the branch-specific receptor assignments.
#     # Its output still contains all 24 receptor channels.
#     # --------------------------------------------------------

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=HISTORY_SIZE,
#         name="neuron_layer_1",
#     )(receptor_input)

#     # --------------------------------------------------------
#     # Neuron layers 2-4
#     #
#     # No input masks here.
#     #
#     # Each layer receives the complete output of the previous
#     # layer, with all receptor channels available.
#     # --------------------------------------------------------

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=HISTORY_SIZE,
#         name="neuron_layer_2",
#     )(x)

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=HISTORY_SIZE,
#         name="neuron_layer_3",
#     )(x)

#     x = neuron.NeuronLIKE(
#         num_neurons=NUM_NEURONS,
#         history_size=HISTORY_SIZE,
#         name="neuron_layer_4",
#     )(x)

#     # --------------------------------------------------------
#     # Conventional classification head
#     # --------------------------------------------------------

#     x = layers.Flatten(
#         name="neuron_output_flatten",
#     )(x)

#     x = layers.Dense(
#         units=32,
#         activation="relu",
#         name="output_hidden",
#     )(x)

#     output = layers.Dense(
#         units=NUM_CLASSES,
#         activation="softmax",
#         name="classification_output",
#     )(x)

#     model = Model(
#         inputs=image,
#         outputs=output,
#         name="hybrid_neuron_network",
#     )

#     return model
# ============================================================
# Three-CNN receptor assignments
# ============================================================

CNN1_RECEPTORS = [
    "AMPA",
    "Kainate",
    "D1",
    "alpha1A",
]

CNN2_RECEPTORS = [
    "NMDA",
    "mGluR1",
    "5-HT2A",
    "H1",
    "H2",
]

CNN3_RECEPTORS = [
    "GABA_A",
    "GABA_B",
    "mGluR2",
    "5-HT1A",
    "D2",
    "CB1",
    "GlyR alpha1",
    "alpha2A",
    "sigma1",
    "MT1",
]


# ============================================================
# Validate three-branch receptor assignments
# ============================================================

def validate_three_cnn_assignments():

    available = set(RECEPTOR_NAMES)

    branches = {
        "CNN1": CNN1_RECEPTORS,
        "CNN2": CNN2_RECEPTORS,
        "CNN3": CNN3_RECEPTORS,
    }

    used = set()

    for branch_name, receptor_names in branches.items():

        if not receptor_names:
            raise ValueError(
                f"{branch_name} has no assigned receptors."
            )

        unknown = set(receptor_names) - available

        if unknown:
            raise ValueError(
                f"{branch_name} contains unknown receptors: "
                f"{sorted(unknown)}"
            )

        overlap = used & set(receptor_names)

        if overlap:
            raise ValueError(
                f"{branch_name} shares receptors with another "
                f"branch: {sorted(overlap)}"
            )

        used.update(receptor_names)

    unused = available - used

    print("\nThree-CNN receptor configuration")
    print("=" * 60)

    for branch_name, receptor_names in branches.items():

        print(f"\n{branch_name} ({len(receptor_names)} receptors):")

        for receptor_name in receptor_names:
            print(f"  + {receptor_name}")

    print(f"\nUnused input channels ({len(unused)}):")

    for receptor_name in RECEPTOR_NAMES:
        if receptor_name in unused:
            print(f"  - {receptor_name}")

    print("\nAll CNN branches have disjoint receptor assignments.")


# ============================================================
# Network construction
# ============================================================

def build_hybrid_network():

    validate_three_cnn_assignments()

    image = layers.Input(
        shape=INPUT_SHAPE,
        name="image",
    )

    # ========================================================
    # CNN 1: Small receptive fields
    # ========================================================

    cnn1 = layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn1_conv_1",
    )(image)

    cnn1 = layers.MaxPooling2D(
        pool_size=2,
        name="cnn1_pool",
    )(cnn1)

    cnn1 = layers.Conv2D(
        filters=24,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn1_conv_2",
    )(cnn1)

    cnn1 = layers.GlobalAveragePooling2D(
        name="cnn1_global_pool",
    )(cnn1)

    # ========================================================
    # CNN 2: Deeper feature extraction
    # ========================================================

    cnn2 = layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn2_conv_1",
    )(image)

    cnn2 = layers.Conv2D(
        filters=24,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn2_conv_2",
    )(cnn2)

    cnn2 = layers.MaxPooling2D(
        pool_size=2,
        name="cnn2_pool",
    )(cnn2)

    cnn2 = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn2_conv_3",
    )(cnn2)

    cnn2 = layers.GlobalAveragePooling2D(
        name="cnn2_global_pool",
    )(cnn2)

    # ========================================================
    # CNN 3: Larger spatial filters
    # ========================================================

    cnn3 = layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="cnn3_conv_1",
    )(image)

    cnn3 = layers.MaxPooling2D(
        pool_size=2,
        name="cnn3_pool",
    )(cnn3)

    cnn3 = layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="cnn3_conv_2",
    )(cnn3)

    cnn3 = layers.GlobalAveragePooling2D(
        name="cnn3_global_pool",
    )(cnn3)

    # ========================================================
    # Project each CNN into the full receptor representation
    # ========================================================
    #
    # Each projection generates:
    #
    #     (batch, NUM_NEURONS, N_RECEPTORS)
    #
    # Independent projections mean that each CNN learns its
    # own receptor signal strengths.
    # ========================================================

    def project_to_receptors(features, prefix):

        signals = layers.Dense(
            units=NUM_NEURONS * N_RECEPTORS,
            activation="sigmoid",
            name=f"{prefix}_receptor_projection",
        )(features)

        signals = layers.Reshape(
            (NUM_NEURONS, N_RECEPTORS),
            name=f"{prefix}_receptor_reshape",
        )(signals)

        return signals

    cnn1_signals = project_to_receptors(
        cnn1,
        "cnn1",
    )

    cnn2_signals = project_to_receptors(
        cnn2,
        "cnn2",
    )

    cnn3_signals = project_to_receptors(
        cnn3,
        "cnn3",
    )

    # ========================================================
    # Route each CNN to its own receptor set
    # ========================================================

    cnn1_signals = ReceptorMask(
        make_receptor_mask(CNN1_RECEPTORS),
        name="cnn1_receptor_routing",
    )(cnn1_signals)

    cnn2_signals = ReceptorMask(
        make_receptor_mask(CNN2_RECEPTORS),
        name="cnn2_receptor_routing",
    )(cnn2_signals)

    cnn3_signals = ReceptorMask(
        make_receptor_mask(CNN3_RECEPTORS),
        name="cnn3_receptor_routing",
    )(cnn3_signals)

    # ========================================================
    # Combine the three branches
    # ========================================================
    #
    # Because the receptor sets are disjoint, each assigned
    # channel receives signals from exactly one CNN branch.
    #
    # Unassigned channels are zero at this point.
    #
    # Output shape:
    # (batch, NUM_NEURONS, N_RECEPTORS)
    # ========================================================

    receptor_input = layers.Add(
        name="three_cnn_receptor_mixer",
    )([
        cnn1_signals,
        cnn2_signals,
        cnn3_signals,
    ])

    # ========================================================
    # Neuron layer 1
    # ========================================================

    x = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
        name="neuron_layer_1",
    )(receptor_input)

    # ========================================================
    # Neuron layers 2-4
    # ========================================================
    #
    # No receptor masks are applied here.
    # Every layer receives the full receptor representation.
    # ========================================================

    x = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
        name="neuron_layer_2",
    )(x)

    x = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
        name="neuron_layer_3",
    )(x)

    x = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
        name="neuron_layer_4",
    )(x)

    # ========================================================
    # Classification head
    # ========================================================

    x = layers.Flatten(
        name="neuron_output_flatten",
    )(x)

    x = layers.Dense(
        units=32,
        activation="relu",
        name="output_hidden",
    )(x)

    output = layers.Dense(
        units=NUM_CLASSES,
        activation="softmax",
        name="classification_output",
    )(x)

    model = Model(
        inputs=image,
        outputs=output,
        name="three_cnn_hybrid_neuron_network",
    )

    return model

# ============================================================
# Inspect trainable neuron parameters
# ============================================================

def inspect_neuron_parameters(model):

    print("\nNeuron parameter statistics")
    print("=" * 70)

    for layer in model.layers:

        if not isinstance(layer, neuron.NeuronLIKE):
            continue

        print(f"\n{layer.name}")

        for parameter_name in (
            "Q_in",
            "Q_out",
            "threshold",
        ):
            parameter = getattr(
                layer,
                parameter_name,
            ).numpy()

            print(
                f"  {parameter_name:10s}: "
                f"min={parameter.min():.4f}, "
                f"mean={parameter.mean():.4f}, "
                f"max={parameter.max():.4f}, "
                f"std={parameter.std():.4f}"
            )

        q_in = layer.Q_in.numpy()
        q_out = layer.Q_out.numpy()
        thresholds = layer.threshold.numpy()

        print("\n  Per-receptor statistics:")

        for receptor_index, receptor_name in enumerate(
            RECEPTOR_NAMES
        ):
            print(
                f"    {receptor_name:15s} | "
                f"Q_in={q_in[:, receptor_index].mean():.4f} | "
                f"Q_out={q_out[:, receptor_index].mean():.4f} | "
                f"threshold="
                f"{thresholds[:, receptor_index].mean():.4f}"
            )


# ============================================================
# Run experiment
# ============================================================

def main():

    # Make experiments easier to reproduce.
    np.random.seed(42)
    tf.random.set_seed(42)

    print("TensorFlow:", tf.__version__)
    print("Receptors:", N_RECEPTORS)
    print("Neurons per layer:", NUM_NEURONS)
    print("Neuron layers: 4")

    # --------------------------------------------------------
    # Build and compile model
    # --------------------------------------------------------

    model = build_hybrid_network()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # --------------------------------------------------------
    # Load MNIST
    # --------------------------------------------------------

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize to [0, 1].
    x_train = (
        x_train.astype(np.float32) / 255.0
    )[..., np.newaxis]

    x_test = (
        x_test.astype(np.float32) / 255.0
    )[..., np.newaxis]
# Select a reproducible random subset of MNIST training data.
    rng = np.random.default_rng(42)

    train_indices = rng.permutation(
        len(x_train)
    )[:TRAIN_SAMPLES]

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    # # Use a subset for this experiment.
    # x_train = x_train[:TRAIN_SAMPLES]
    # y_train = y_train[:TRAIN_SAMPLES]

    x_test = x_test[:TEST_SAMPLES]
    y_test = y_test[:TEST_SAMPLES]

    print("\nDataset")
    print("=" * 60)
    print("Training examples:", len(x_train))
    print("Test examples:", len(x_test))
    print("Input shape:", x_train.shape[1:])
    print("Classes:", NUM_CLASSES)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,
    )

    # --------------------------------------------------------
    # Inspect learned neuron parameters
    # --------------------------------------------------------

    inspect_neuron_parameters(model)

    # --------------------------------------------------------
    # Evaluate on held-out test subset
    # --------------------------------------------------------

    print("\nEvaluating...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # --------------------------------------------------------
    # Show a few predictions
    # --------------------------------------------------------

    predictions = model.predict(
        x_test[:5],
        verbose=0,
    )

    print("\nSample predictions")
    print("=" * 60)

    for i, probabilities in enumerate(predictions):

        predicted_class = int(
            np.argmax(probabilities)
        )

        actual_class = int(y_test[i])

        confidence = float(
            np.max(probabilities)
        )

        print(
            f"Image {i}: "
            f"predicted={predicted_class}, "
            f"actual={actual_class}, "
            f"confidence={confidence:.3f}"
        )

    # --------------------------------------------------------
    # Save complete model
    # --------------------------------------------------------

    model.save("hybrid_neuron_net.keras")

    print("\nSaved: hybrid_neuron_net.keras")


if __name__ == "__main__":
    main()
