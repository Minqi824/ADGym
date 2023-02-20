# ADGym

ADGym is able to applied to unsupervised (to do), semi-/weakly-supervised and fully-supervised scenarios.
Currently, ADGym is mainly devised for the **tabular** data.

ADGym considers multiple components in each part of the following pipeline:  
**Data Augmentation** → **Data Preprocessing** → **Network Architecture** → **Network Training**  

Each part of the pipeline can be instantiated by multiple components:
| Pipeline | Detailed Components | Value |
|:--:|:--:|:--:|
|Data Augmentation||[Oversampling, SMOTE, Mixup, GAN]|


### Data Augmentation
- Oversampling
- SMOTE
- Mixup
- GAN

### Data Preprocessing
- MinMax
- Normalization

### Network Architecture
- MLP
- AutoEncoder
- ResNet
- FTTransformer

### Network Training
- default (pytorch default)
- Xavier (uniform)
- Xavier (normal)
- Kaiming (uniform)
- Kaiming (normal)

### Network Hyperparameter
- neurons
- hidden layers
- batch normalization
- activation layer
- dropout

### Loss Function
- Cross Entropy
- Focal loss
- Minus loss
- Inverse loss
- Hinge loss
- Deviation loss
- Ordinal loss
- ...

### Training Strategy
- batch resampling
- ...

## Python Package Requirements
- iteration_utilities==0.11.0

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