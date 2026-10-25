"""
Comprehensive integration tests for neuron_project_fixed.py.

Tests:
    1. Python, NumPy, TensorFlow, and Keras environment.
    2. Receptor definitions and interaction matrix consistency.
    3. Invalid-input handling.
    4. NeuronLIKE forward pass and output shapes.
    5. Gradient propagation and numerical stability.
    6. TensorFlow graph execution with tf.function.
    7. History-buffer updates.
    8. Actual Keras training with model.fit().
    9. Weight updates and Q-factor constraints.
   10. Model evaluation and prediction.
   11. Full Keras model serialization and restoration.
   12. Custom-layer weight serialization.

Run:
    python test_neuron_project.py

Dependencies:
    numpy
    tensorflow
"""

from pathlib import Path
import tempfile
import traceback

import numpy as np
import tensorflow as tf

import neuron_like as neuron


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 8
NUM_NEURONS = 3
NUM_LAYERS = 2
HISTORY_SIZE = 16

NUM_TRAIN_SAMPLES = 64
NUM_TRAINING_EPOCHS = 3

LEARNING_RATE = 1e-3

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Test utilities
# ============================================================

passed = []
failed = []


def run_test(name, function):
    """Run one test and report its result without hiding failures."""

    print(f"\n[TEST] {name}")

    try:
        function()
        passed.append(name)
        print(f"[PASS] {name}")

    except Exception as exc:
        failed.append((name, exc))
        print(f"[FAIL] {name}")
        print(f"       {type(exc).__name__}: {exc}")


def check(condition, message):
    """Raise an exception if a test condition is false."""

    if not condition:
        raise AssertionError(message)


def check_finite(tensor, name):
    """Check that every element of a tensor is finite."""

    tensor = tf.convert_to_tensor(tensor)

    check(
        bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()),
        f"{name} contains NaN or infinity.",
    )


def check_q_constraints(layer):
    """Check Q_in and Q_out against the project's [0.1, 10] limits."""

    for name in ("Q_in", "Q_out"):
        value = getattr(layer, name)

        check_finite(value, name)

        check(
            bool(tf.reduce_all(value >= 0.1).numpy()),
            f"{name} contains values below 0.1.",
        )

        check(
            bool(tf.reduce_all(value <= 10.0).numpy()),
            f"{name} contains values above 10.0.",
        )


def make_model():
    """Construct and build a fresh project model."""

    model = neuron.NeuronNetwork(
        num_neurons=NUM_NEURONS,
        num_layers=NUM_LAYERS,
        history_size=HISTORY_SIZE,
    )

    dummy_input = tf.zeros(
        (1, NUM_NEURONS, neuron.N_RECEPTORS),
        dtype=tf.float32,
    )

    model(dummy_input, training=False)

    return model


# ============================================================
# Test 1: Environment
# ============================================================

def test_environment():
    print("Python module:", neuron.__file__)
    print("TensorFlow:", tf.__version__)
    print(
        "Keras:",
        getattr(tf.keras, "__version__", "bundled with TensorFlow"),
    )
    print("NumPy:", np.__version__)
    print("Receptor count:", neuron.N_RECEPTORS)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    check(
        neuron.N_RECEPTORS == len(neuron.RECEPTOR_NAMES),
        "Receptor count does not match receptor names.",
    )

    check(
        neuron.N_RECEPTORS > 0,
        "No receptors are defined.",
    )


# ============================================================
# Test 2: Receptor rules and interaction matrix
# ============================================================

def test_receptor_rules():
    neuron.validate_receptor_rules(neuron.RECEPTOR_RULES)

    names = neuron.RECEPTOR_NAMES
    matrix = neuron.INTERACTION_MATRIX

    check(
        len(names) == neuron.N_RECEPTORS,
        "Incorrect receptor-name count.",
    )

    check(
        len(set(names)) == len(names),
        "Duplicate receptor names.",
    )

    check(
        matrix.shape == (
            neuron.N_RECEPTORS,
            neuron.N_RECEPTORS,
        ),
        f"Unexpected matrix shape: {matrix.shape}",
    )

    check(
        np.isfinite(matrix).all(),
        "Interaction matrix contains NaN or infinity.",
    )

    check(
        np.isin(matrix, [-1.0, 0.0, 1.0]).all(),
        "Interaction matrix contains unexpected values.",
    )

    # Verify every defined interaction is represented correctly.
    index = {
        name: i
        for i, name in enumerate(names)
    }

    for target, rules in neuron.RECEPTOR_RULES.items():

        target_i = index[target]

        for source in rules["stimuli"]:
            source_i = index[source]

            check(
                matrix[target_i, source_i] == 1.0,
                f"Missing stimulation edge: {source} -> {target}",
            )

        for source in rules["inhibitory"]:
            source_i = index[source]

            check(
                matrix[target_i, source_i] == -1.0,
                f"Missing inhibitory edge: {source} -> {target}",
            )

    # Explicitly verify the corrected mGluR2 rules.
    mglur2 = neuron.RECEPTOR_RULES["mGluR2"]

    check(
        mglur2["stimuli"] == [],
        "Unexpected mGluR2 stimuli.",
    )

    check(
        set(mglur2["inhibitory"]) == {"AMPA", "NMDA"},
        "Unexpected mGluR2 inhibitory connections.",
    )


# ============================================================
# Test 3: Single neuron-layer forward pass
# ============================================================

def test_single_layer():
    layer = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
    )

    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    y = layer(x, training=False)

    expected_shape = (
        BATCH_SIZE,
        NUM_NEURONS,
        neuron.N_RECEPTORS,
    )

    check(
        tuple(y.shape) == expected_shape,
        f"Wrong output shape: {y.shape}",
    )

    check_finite(y, "Single-layer output")
    check_q_constraints(layer)

    check(
        tuple(layer.get_history_mean().shape)
        == (NUM_NEURONS, neuron.N_RECEPTORS),
        "Incorrect history-mean shape.",
    )


# ============================================================
# Test 4: Invalid-input handling
# ============================================================

def test_invalid_inputs():
    layer = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
    )

    # Build the layer with valid input first.
    valid = tf.zeros(
        (1, NUM_NEURONS, neuron.N_RECEPTORS),
        dtype=tf.float32,
    )

    layer(valid)

    invalid_inputs = [
        # Wrong neuron count.
        tf.zeros(
            (1, NUM_NEURONS + 1, neuron.N_RECEPTORS)
        ),

        # Wrong receptor count.
        tf.zeros(
            (1, NUM_NEURONS, neuron.N_RECEPTORS + 1)
        ),

        # Wrong rank.
        tf.zeros(
            (NUM_NEURONS, neuron.N_RECEPTORS)
        ),
    ]

    for i, bad_input in enumerate(invalid_inputs):
        rejected = False

        try:
            layer(bad_input, training=False)

        except (ValueError, tf.errors.InvalidArgumentError):
            rejected = True

        check(
            rejected,
            f"Invalid input #{i + 1} was not rejected.",
        )


# ============================================================
# Test 5: Gradient propagation
# ============================================================

def test_gradients():
    model = make_model()

    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    target = tf.random.uniform(
        tf.shape(x),
        minval=0.0,
        maxval=1.0,
    )

    with tf.GradientTape() as tape:
        prediction = model(x, training=False)

        loss = tf.reduce_mean(
            tf.square(prediction - target)
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )

    check(
        len(gradients) == len(model.trainable_variables),
        "Gradient count does not match trainable-variable count.",
    )

    disconnected = []
    non_finite = []

    for variable, gradient in zip(
        model.trainable_variables,
        gradients,
    ):
        if gradient is None:
            disconnected.append(variable.name)

        elif not bool(
            tf.reduce_all(
                tf.math.is_finite(gradient)
            ).numpy()
        ):
            non_finite.append(variable.name)

    check(
        not disconnected,
        f"Variables without gradients: {disconnected}",
    )

    check(
        not non_finite,
        f"Variables with non-finite gradients: {non_finite}",
    )

    check(
        bool(tf.math.is_finite(loss).numpy()),
        "Loss is not finite.",
    )

    print("Initial gradient-test loss:", float(loss.numpy()))
    print("Trainable variables:", len(model.trainable_variables))


# ============================================================
# Test 6: TensorFlow graph execution
# ============================================================

def test_tf_function():
    model = make_model()

    @tf.function
    def inference_step(inputs):
        return model(inputs, training=False)

    @tf.function
    def training_step(inputs):
        return model(inputs, training=True)

    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    # Test graph-compiled inference.
    y_inference = inference_step(x)

    check(
        tuple(y_inference.shape)
        == (
            BATCH_SIZE,
            NUM_NEURONS,
            neuron.N_RECEPTORS,
        ),
        "tf.function inference returned the wrong shape.",
    )

    check_finite(y_inference, "Graph inference output")

    # Test graph-compiled execution with training=True.
    y_training = training_step(x)

    check_finite(y_training, "Graph training output")

    # The training path should update the history buffer.
    for layer in model.neuron_layers:
        check(
            int(layer._history_ptr.numpy()) > 0,
            f"{layer.name} did not update its history buffer.",
        )


# ============================================================
# Test 7: Actual TensorFlow/Keras training
# ============================================================

def test_keras_training():
    model = make_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    # Synthetic inputs and regression targets.
    # These test software compatibility, not biological accuracy.
    rng = np.random.default_rng(SEED)

    x = rng.uniform(
        0.0,
        1.0,
        size=(
            NUM_TRAIN_SAMPLES,
            NUM_NEURONS,
            neuron.N_RECEPTORS,
        ),
    ).astype(np.float32)

    y = rng.uniform(
        0.0,
        1.0,
        size=x.shape,
    ).astype(np.float32)

    # Snapshot trainable variables before training.
    weights_before = [
        variable.numpy().copy()
        for variable in model.trainable_variables
    ]

    # Exercise the real Keras training loop.
    history = model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=NUM_TRAINING_EPOCHS,
        shuffle=False,
        verbose=0,
    )

    check(
        "loss" in history.history,
        "Keras training did not report loss.",
    )

    losses = np.asarray(
        history.history["loss"],
        dtype=np.float64,
    )

    check(
        len(losses) == NUM_TRAINING_EPOCHS,
        "Unexpected number of training epochs.",
    )

    check(
        np.isfinite(losses).all(),
        f"Training produced invalid losses: {losses}",
    )

    # Check that optimization actually changed parameters.
    weights_after = [
        variable.numpy()
        for variable in model.trainable_variables
    ]

    changed = any(
        not np.array_equal(before, after)
        for before, after in zip(
            weights_before,
            weights_after,
        )
    )

    check(
        changed,
        "Training did not update any trainable variables.",
    )

    # Verify Q-factor constraints after optimizer updates.
    for layer in model.neuron_layers:
        check_q_constraints(layer)

    # Verify history-buffer state changed during training.
    for layer in model.neuron_layers:
        check(
            int(layer._history_ptr.numpy()) > 0,
            f"{layer.name} history did not update during fit().",
        )

    print("Training losses:", losses.tolist())
    print("Final loss:", float(losses[-1]))

    # Store the trained model for later tests.
    global trained_model, training_x

    trained_model = model
    training_x = x


# ============================================================
# Test 8: Evaluation and prediction
# ============================================================

def test_evaluate_predict():
    model = make_model()

    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    y = tf.random.uniform(
        tf.shape(x),
        minval=0.0,
        maxval=1.0,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="mse",
    )

    # One training batch.
    train_result = model.train_on_batch(x, y)

    train_loss = (
        train_result[0]
        if isinstance(train_result, (list, tuple))
        else train_result
    )

    check(
        np.isfinite(float(train_loss)),
        "train_on_batch() returned an invalid loss.",
    )

    # Evaluation.
    eval_result = model.evaluate(
        x,
        y,
        verbose=0,
    )

    eval_loss = (
        eval_result[0]
        if isinstance(eval_result, (list, tuple))
        else eval_result
    )

    check(
        np.isfinite(float(eval_loss)),
        "evaluate() returned an invalid loss.",
    )

    # Prediction.
    predictions = model.predict(
        x,
        verbose=0,
    )

    check(
        predictions.shape == x.shape,
        f"Unexpected prediction shape: {predictions.shape}",
    )

    check_finite(predictions, "Keras predictions")


# ============================================================
# Test 9: Full Keras model save/load
# ============================================================

def test_keras_serialization():
    model = make_model()

    # Give the model at least one optimizer update before saving.
    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    y = tf.random.uniform(
        tf.shape(x),
        minval=0.0,
        maxval=1.0,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="mse",
    )

    model.train_on_batch(x, y)

    # Use inference mode so history changes do not interfere with comparison.
    expected = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "neuron_model.keras"

        model.save(path)

        check(
            path.exists() and path.stat().st_size > 0,
            "Keras model file was not created.",
        )

        # The module has already been imported, registering custom classes.
        restored = tf.keras.models.load_model(
            path,
            compile=True,
        )

        actual = restored(x, training=False).numpy()

        check(
            isinstance(restored, neuron.NeuronNetwork),
            f"Unexpected restored type: {type(restored)}",
        )

        check(
            actual.shape == expected.shape,
            "Restored model has a different output shape.",
        )

        check(
            np.allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
            ),
            "Restored model output differs from the original.",
        )

        # Verify the restored model can still train.
        result = restored.train_on_batch(x, y)

        loss = (
            result[0]
            if isinstance(result, (list, tuple))
            else result
        )

        check(
            np.isfinite(float(loss)),
            "Restored model cannot train correctly.",
        )


# ============================================================
# Test 10: Pickle-based layer-weight save/load
# ============================================================

def test_layer_weight_serialization():
    x = tf.random.uniform(
        (BATCH_SIZE, NUM_NEURONS, neuron.N_RECEPTORS),
        minval=0.0,
        maxval=1.0,
    )

    original = neuron.NeuronLIKE(
        num_neurons=NUM_NEURONS,
        history_size=HISTORY_SIZE,
    )

    original(x, training=False)

    # Assign distinctive but valid values to test exact restoration.
    original.Q_in.assign(
        tf.ones_like(original.Q_in) * 1.25
    )

    original.Q_out.assign(
        tf.ones_like(original.Q_out) * 0.75
    )

    original.threshold.assign(
        tf.ones_like(original.threshold) * 0.2
    )

    expected = original(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "neuron_weights.pkl"

        original.save_weights_pickle(path)

        restored = neuron.NeuronLIKE(
            num_neurons=NUM_NEURONS,
            history_size=HISTORY_SIZE,
        )

        restored(x, training=False)
        restored.load_weights_pickle(path)

        actual = restored(x, training=False).numpy()

        check(
            np.allclose(
                expected,
                actual,
                rtol=1e-6,
                atol=1e-7,
            ),
            "Layer weights did not round-trip correctly.",
        )

        check_q_constraints(restored)


# ============================================================
# Run everything
# ============================================================

def main():
    print("=" * 65)
    print("NEURON PROJECT: INTEGRATION TESTS")
    print("=" * 65)

    tests = [
        ("Environment and dependencies", test_environment),

        ("Receptor rules and interaction matrix", test_receptor_rules),

        ("NeuronLIKE forward pass", test_single_layer),

        ("Invalid-input handling", test_invalid_inputs),

        ("Gradient propagation", test_gradients),

        ("TensorFlow tf.function execution", test_tf_function),

        ("Keras training with model.fit()", test_keras_training),

        ("Training, evaluation, and prediction APIs", test_evaluate_predict),

        ("Full Keras save/load compatibility", test_keras_serialization),

        ("Layer-weight save/load compatibility", test_layer_weight_serialization),
    ]

    for name, function in tests:
        run_test(name, function)

    print("\n" + "=" * 65)
    print("TEST SUMMARY")
    print("=" * 65)

    print(f"Passed: {len(passed)} / {len(tests)}")
    print(f"Failed: {len(failed)} / {len(tests)}")

    if failed:
        print("\nFAILURES:")

        for name, exc in failed:
            print(f"\n--- {name} ---")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )

        raise SystemExit(1)

    print("\nALL INTEGRATION TESTS PASSED.")


if __name__ == "__main__":
    main()
