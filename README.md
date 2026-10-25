# Advanced-Perceptron-Project
tensorflow - keras layer, that add colors to the inputs and outputs

# Neuron APP - Advanced Perceptron Project
ML tool for various tasks - to be continued

1. PROJECT DESCRIPTION

Perceptron much more biological neuron-like than perceptrons in current use. Within one graph node several networks are connected interacting with each other - within built-in strict rules. And yes...

One neuron from this project CAN be XOR gate.

2. WHAT IS INSIDE

Well. Each neuron is just like perceptrons - multiple just in one node. Those perceptrons inside one neuron can interact with each other, influencing/altering their action. Look:
    
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

These above are just main transmitters and their receptors - the first part to be aware of. Some of the receptors are altering function of others - negatively and positively, depending whether they are stimuli/inhibitory. Now the more funny part - the way, how all of those nets are encoded to interact with each other inside single AP - advanced perceptron:

So, we have several main networks just in one neuron, and available subnets are the receptor types. Those perceptrons are interconnected inside of neuron (node) WITH STRICT RULES. Example: noradrenaline (alpha receptors):

        "alpha1A": {
            "stimuli": ["AMPA"],
            "inhibitory": [],
        },
        "alpha2A": {
            "stimuli": [],
            "inhibitory": ["D1", "5-HT2A"],
        },
            
Stimuli and inhibitory means that the floating point numbers, inputs of noradrenaline network will be increased by ampa, and decreased by d1 and 5-ht2a - not only activation function is taken into account.

So: assume, alpha1A has 2.0 on input. Theres also AMPA with value of 3, and D1 with value of 4. It means, that norepinephrine final output will be smaller (!) than after simple activation function - D1 (inhibitor) has bigger value than AMPA stimuli.

Therefore - each single neuron can act as XOR gate (and even more).

Whats important - the degree of influence by other networks adjusts during training.

Each neuron has memory, denoted by 'history' variable. History is just a list of previous transmitter values, and it take part in learning process, introducing hysteresis.
In real world cells, memory can be at DNA level - for example if g-protein activated certain genes, altering transmitter production at output. It can be also increase/decrease in receptor numbers at input, due to sensitization/tolerance developement. 

Here number of receptors is stable, but their response differs. It would be nice to code net with floating internal structure over time - not predefined like here. I hope I ll do it some day.

Loss is very important - it is the info about difference between desired state and the goal to achieve. Just like in robots - Loss can be just the difference between sensor reading (current) and the desired state. The net 'll learn to achieve that. To compensate non-desired state of the system.

3. HOW TO USE THIS

Make complex networks. Add different keras networks (dense, conv, etc) to different receptor types, or split output of one keras net feeding different receptor types. those receptors add colors to the grayscale of floats and ints. 

typical output after 20 epochs (note higher val stats, dataset needs some tweaks):


 Total params: 630,718 (2.41 MB)
 Trainable params: 235,194 (918.73 KB)
 Non-trainable params: 395,524 (1.51 MB)

Dataset
============================================================
Training examples: 40000
Test examples: 10000
Input shape: (28, 28, 1)
Classes: 10

Starting training...
Epoch 1/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 31s 50ms/step - accuracy: 0.1694 - loss: 2.1776 - val_accuracy: 0.2027 - val_loss: 2.0421
Epoch 2/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 49ms/step - accuracy: 0.2168 - loss: 2.0039 - val_accuracy: 0.1980 - val_loss: 1.9693
Epoch 3/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.2253 - loss: 1.9125 - val_accuracy: 0.1968 - val_loss: 1.8521
Epoch 4/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 29s 52ms/step - accuracy: 0.2439 - loss: 1.7948 - val_accuracy: 0.2262 - val_loss: 1.7377
Epoch 5/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 28s 49ms/step - accuracy: 0.3084 - loss: 1.6620 - val_accuracy: 0.3295 - val_loss: 1.5788
Epoch 6/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 49ms/step - accuracy: 0.3188 - loss: 1.5797 - val_accuracy: 0.3322 - val_loss: 1.5271
Epoch 7/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 49ms/step - accuracy: 0.3345 - loss: 1.5293 - val_accuracy: 0.3435 - val_loss: 1.4692
Epoch 8/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.3731 - loss: 1.4608 - val_accuracy: 0.3952 - val_loss: 1.3897
Epoch 9/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 49ms/step - accuracy: 0.4263 - loss: 1.3577 - val_accuracy: 0.4925 - val_loss: 1.2147
Epoch 10/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 28s 50ms/step - accuracy: 0.4873 - loss: 1.1879 - val_accuracy: 0.5130 - val_loss: 1.1186
Epoch 11/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 28s 50ms/step - accuracy: 0.5167 - loss: 1.1155 - val_accuracy: 0.5410 - val_loss: 1.0601
Epoch 12/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 28s 49ms/step - accuracy: 0.5484 - loss: 1.0536 - val_accuracy: 0.5688 - val_loss: 1.0221
Epoch 13/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.5867 - loss: 0.9932 - val_accuracy: 0.6030 - val_loss: 0.9517
Epoch 14/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.6718 - loss: 0.9045 - val_accuracy: 0.7448 - val_loss: 0.8157
Epoch 15/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 49ms/step - accuracy: 0.7378 - loss: 0.7955 - val_accuracy: 0.7785 - val_loss: 0.7148
Epoch 16/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.7764 - loss: 0.6475 - val_accuracy: 0.8602 - val_loss: 0.4602
Epoch 17/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.8821 - loss: 0.4053 - val_accuracy: 0.9147 - val_loss: 0.2994
Epoch 18/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.9071 - loss: 0.3252 - val_accuracy: 0.9262 - val_loss: 0.2573
Epoch 19/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 29s 52ms/step - accuracy: 0.9212 - loss: 0.2780 - val_accuracy: 0.9325 - val_loss: 0.2371
Epoch 20/20
563/563 ━━━━━━━━━━━━━━━━━━━━ 27s 48ms/step - accuracy: 0.9294 - loss: 0.2501 - val_accuracy: 0.9337 - val_loss: 0.2221

Neuron parameter statistics
======================================================================

neuron_layer_1
  Q_in      : min=0.8378, mean=1.0050, max=1.2230, std=0.0622
  Q_out     : min=0.8693, mean=1.0186, max=1.6545, std=0.0864
  threshold : min=-0.5971, mean=-0.0179, max=0.1009, std=0.0651

  Per-receptor statistics:
    AMPA            | Q_in=0.9968 | Q_out=1.1294 | threshold=-0.1242
    NMDA            | Q_in=0.9787 | Q_out=1.0519 | threshold=-0.0549
    Kainate         | Q_in=1.0090 | Q_out=1.0128 | threshold=0.0000
    mGluR1          | Q_in=0.9979 | Q_out=1.0136 | threshold=-0.0012
    mGluR2          | Q_in=1.0054 | Q_out=0.9999 | threshold=0.0000
    GABA_A          | Q_in=1.0106 | Q_out=1.0235 | threshold=-0.0238
    GABA_B          | Q_in=1.0054 | Q_out=1.0054 | threshold=-0.0040
    D1              | Q_in=1.0028 | Q_out=0.9990 | threshold=0.0083
    D2              | Q_in=1.0214 | Q_out=1.0093 | threshold=-0.0173
    5-HT1A          | Q_in=1.0177 | Q_out=1.1258 | threshold=-0.1257
    5-HT2A          | Q_in=0.9998 | Q_out=1.0223 | threshold=-0.0153
    5-HT3           | Q_in=1.0161 | Q_out=0.9954 | threshold=0.0000
    H1              | Q_in=0.9904 | Q_out=1.1519 | threshold=-0.1546
    H2              | Q_in=1.0790 | Q_out=0.9856 | threshold=0.0167
    H3              | Q_in=1.0053 | Q_out=0.9935 | threshold=0.0000
    CB1             | Q_in=0.9941 | Q_out=0.9847 | threshold=0.0128
    Mu (mi)         | Q_in=1.0078 | Q_out=1.0116 | threshold=-0.0075
    GlyR alpha1     | Q_in=0.9772 | Q_out=0.9964 | threshold=0.0154
    alpha1A         | Q_in=0.9998 | Q_out=0.9922 | threshold=0.0152
    alpha2A         | Q_in=1.0178 | Q_out=0.9980 | threshold=0.0000
    sigma1          | Q_in=0.9801 | Q_out=0.9834 | threshold=0.0154
    MT1             | Q_in=0.9981 | Q_out=0.9655 | threshold=0.0293
    OX1             | Q_in=1.0105 | Q_out=0.9906 | threshold=0.0000
    NK1             | Q_in=0.9986 | Q_out=1.0052 | threshold=-0.0149

neuron_layer_2
  Q_in      : min=0.8109, mean=1.0198, max=1.5863, std=0.0879
  Q_out     : min=0.8476, mean=1.0397, max=1.8102, std=0.1033
  threshold : min=-0.8577, mean=-0.0371, max=0.1283, std=0.0868

  Per-receptor statistics:
    AMPA            | Q_in=1.1271 | Q_out=1.0186 | threshold=0.0080
    NMDA            | Q_in=1.0616 | Q_out=1.0098 | threshold=0.0247
    Kainate         | Q_in=1.0065 | Q_out=0.9992 | threshold=0.0000
    mGluR1          | Q_in=1.0107 | Q_out=1.0703 | threshold=-0.0601
    mGluR2          | Q_in=1.0035 | Q_out=0.9963 | threshold=0.0000
    GABA_A          | Q_in=1.0216 | Q_out=1.0493 | threshold=-0.0143
    GABA_B          | Q_in=1.0051 | Q_out=1.0324 | threshold=-0.0492
    D1              | Q_in=1.0026 | Q_out=1.1025 | threshold=-0.1053
    D2              | Q_in=1.0255 | Q_out=1.0143 | threshold=-0.0282
    5-HT1A          | Q_in=1.1221 | Q_out=1.0937 | threshold=-0.0890
    5-HT2A          | Q_in=1.0263 | Q_out=1.0249 | threshold=-0.0108
    5-HT3           | Q_in=1.0070 | Q_out=0.9929 | threshold=0.0000
    H1              | Q_in=1.1513 | Q_out=0.9770 | threshold=0.0308
    H2              | Q_in=0.9774 | Q_out=1.0462 | threshold=-0.0433
    H3              | Q_in=1.0036 | Q_out=0.9946 | threshold=0.0000
    CB1             | Q_in=0.9814 | Q_out=1.1447 | threshold=-0.1648
    Mu (mi)         | Q_in=1.0159 | Q_out=1.0122 | threshold=-0.0193
    GlyR alpha1     | Q_in=0.9800 | Q_out=1.0219 | threshold=-0.0412
    alpha1A         | Q_in=0.9789 | Q_out=1.1134 | threshold=-0.1106
    alpha2A         | Q_in=1.0066 | Q_out=1.0086 | threshold=0.0000
    sigma1          | Q_in=0.9873 | Q_out=1.0423 | threshold=-0.0411
    MT1             | Q_in=0.9621 | Q_out=1.1271 | threshold=-0.1295
    OX1             | Q_in=0.9962 | Q_out=1.0129 | threshold=0.0000
    NK1             | Q_in=1.0154 | Q_out=1.0487 | threshold=-0.0462

neuron_layer_3
  Q_in      : min=0.8316, mean=1.0414, max=1.8011, std=0.1030
  Q_out     : min=0.7570, mean=1.0176, max=1.9152, std=0.1095
  threshold : min=-0.4645, mean=-0.0054, max=0.2369, std=0.0742

  Per-receptor statistics:
    AMPA            | Q_in=1.0337 | Q_out=1.1011 | threshold=-0.0297
    NMDA            | Q_in=1.0024 | Q_out=1.1388 | threshold=-0.0817
    Kainate         | Q_in=1.0039 | Q_out=1.0015 | threshold=0.0000
    mGluR1          | Q_in=1.0796 | Q_out=0.9905 | threshold=0.0373
    mGluR2          | Q_in=0.9953 | Q_out=0.9974 | threshold=0.0000
    GABA_A          | Q_in=1.0343 | Q_out=0.9307 | threshold=0.0741
    GABA_B          | Q_in=1.0255 | Q_out=0.9669 | threshold=0.0351
    D1              | Q_in=1.0938 | Q_out=0.9933 | threshold=0.0579
    D2              | Q_in=1.0206 | Q_out=1.0445 | threshold=-0.0476
    5-HT1A          | Q_in=1.0683 | Q_out=0.9808 | threshold=-0.0008
    5-HT2A          | Q_in=1.0538 | Q_out=1.1371 | threshold=-0.0823
    5-HT3           | Q_in=0.9959 | Q_out=1.0088 | threshold=0.0000
    H1              | Q_in=0.9806 | Q_out=1.1102 | threshold=-0.1030
    H2              | Q_in=1.0330 | Q_out=0.9968 | threshold=0.0201
    H3              | Q_in=0.9984 | Q_out=1.0089 | threshold=0.0000
    CB1             | Q_in=1.1631 | Q_out=0.9672 | threshold=0.0364
    Mu (mi)         | Q_in=1.0211 | Q_out=1.0796 | threshold=-0.0728
    GlyR alpha1     | Q_in=1.0482 | Q_out=0.9618 | threshold=0.0290
    alpha1A         | Q_in=1.1145 | Q_out=0.9915 | threshold=0.0128
    alpha2A         | Q_in=0.9976 | Q_out=1.0036 | threshold=0.0000
    sigma1          | Q_in=1.0500 | Q_out=0.9580 | threshold=0.0289
    MT1             | Q_in=1.1325 | Q_out=0.9786 | threshold=0.0193
    OX1             | Q_in=0.9936 | Q_out=1.0123 | threshold=0.0000
    NK1             | Q_in=1.0549 | Q_out=1.0639 | threshold=-0.0633

neuron_layer_4
  Q_in      : min=0.6659, mean=1.0185, max=1.7717, std=0.1057
  Q_out     : min=0.8312, mean=1.0496, max=1.6678, std=0.1051
  threshold : min=-0.4851, mean=-0.0131, max=0.4672, std=0.1015

  Per-receptor statistics:
    AMPA            | Q_in=1.1024 | Q_out=1.0157 | threshold=0.0592
    NMDA            | Q_in=1.1417 | Q_out=1.0142 | threshold=0.0651
    Kainate         | Q_in=1.0019 | Q_out=1.0056 | threshold=0.0537
    mGluR1          | Q_in=0.9782 | Q_out=1.1097 | threshold=0.0317
    mGluR2          | Q_in=1.0156 | Q_out=1.1250 | threshold=-0.1118
    GABA_A          | Q_in=0.9325 | Q_out=1.1068 | threshold=-0.0892
    GABA_B          | Q_in=0.9522 | Q_out=1.0809 | threshold=-0.0740
    D1              | Q_in=1.0070 | Q_out=1.1177 | threshold=0.0581
    D2              | Q_in=1.0434 | Q_out=1.0660 | threshold=-0.0571
    5-HT1A          | Q_in=0.9911 | Q_out=1.0413 | threshold=-0.0421
    5-HT2A          | Q_in=1.1298 | Q_out=1.0059 | threshold=0.0636
    5-HT3           | Q_in=0.9953 | Q_out=1.0026 | threshold=0.0016
    H1              | Q_in=1.1218 | Q_out=1.0041 | threshold=0.0473
    H2              | Q_in=1.0030 | Q_out=1.0537 | threshold=-0.0097
    H3              | Q_in=0.9962 | Q_out=1.0063 | threshold=0.0086
    CB1             | Q_in=0.9776 | Q_out=1.0941 | threshold=-0.0866
    Mu (mi)         | Q_in=1.0741 | Q_out=1.0703 | threshold=-0.0641
    GlyR alpha1     | Q_in=0.9825 | Q_out=1.0295 | threshold=-0.0435
    alpha1A         | Q_in=1.0016 | Q_out=1.0741 | threshold=-0.0029
    alpha2A         | Q_in=1.0053 | Q_out=1.0552 | threshold=-0.0495
    sigma1          | Q_in=0.9660 | Q_out=1.0485 | threshold=-0.0558
    MT1             | Q_in=0.9761 | Q_out=1.0456 | threshold=-0.0333
    OX1             | Q_in=1.0013 | Q_out=1.0195 | threshold=0.0145
    NK1             | Q_in=1.0484 | Q_out=0.9970 | threshold=0.0027

Evaluating...
157/157 ━━━━━━━━━━━━━━━━━━━━ 2s 13ms/step - accuracy: 0.9403 - loss: 0.2038

Test loss:     0.2038
Test accuracy: 0.9403

Sample predictions
============================================================
Image 0: predicted=7, actual=7, confidence=0.998
Image 1: predicted=2, actual=2, confidence=0.584
Image 2: predicted=1, actual=1, confidence=1.000
Image 3: predicted=0, actual=0, confidence=0.889
Image 4: predicted=4, actual=4, confidence=1.000

Saved: hybrid_neuron_net.keras
(.venv) tk@tk-Aspire-A715-72G:~/NeuronProject$ 
