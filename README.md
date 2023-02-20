# ADGym :running_man:

ADGym is able to applied to unsupervised (to do), semi-/weakly-supervised and fully-supervised scenarios.
Currently, ADGym is mainly devised for the **tabular** data.

ADGym considers multiple components in each part of the following pipeline:  
**Data Augmentation** → **Data Preprocessing** → **Network Architecture** → **Network Training**  

Each part of the pipeline can be instantiated by multiple components (core components are marked in **bold**):
| Pipeline | Detailed Components | Value |
|:--:|:--:|:--:|
|**Data Augmentation**||[Oversampling, SMOTE, Mixup, GAN]|
|Data Preprocessing||[MinMax, Normalization]|
|Network Architecture|**Network Architecture**|[MLP, AutoEncoder, ResNet, FTTransformer]|
||Hidden Layers|[1, 2, 3]|
||Neurons|[[20], [100, 20], [100, 50, 20]]|
||**Activation**|[Tanh, ReLU, LeakyReLU]|
||Dropout|[0.0, 0.1, 0.3]|
||**Initialization**|[PyTorch default, Xavier (uniform), Xavier (normal), Kaiming (uniform), Kaiming (normal)]|
|Network Training|**Loss Function**|[BCE, Focal, Minus, Inverse, Hinge, Deviation, Ordinal]|
||**Optimizer**|[SGD, Adam, RMSprop]|
||**Batch Resampling**|[False, True]|
||**Epochs**|[20, 50, 100]|
||Batch Size|[16, 64, 256]|
||**Learning Rate**|[1e-2, 1e-3]|
||Weight Decay|[1e-2, 1e-4]|

## Quick Start with ADGym
```python
to do...
```

## Python Package Requirements
- iteration_utilities==0.11.0
- metaod==0.0.6
- scikit-learn==0.23.2
- install imbalanced-learn==0.7.0
- torch==1.9.0
- tabgan==1.2.1

## Update Logs
- 2022.11.17: run the experiments of current component combinations
- 2022.11.23: add the GAN-based data augmentation method
- 2022.11.25: add the oversampling and SMOTE data augmentation method
- 2022.11.25: add the binary cross entropy loss and focal loss
- 2023.01.04: add the Mixup data augmentation method
- 2023.01.04: add different network initialization methods
- 2023.01.04: add the ordinal loss in PReNet model
- 2023.01.04: revise the labeled anomalies to the number (instead of ratio) of labeled anomalies
- 2023.02.20: restart ADGym
- 2023.02.20: add two baselines: random selection and model selection based on the partially labeled data