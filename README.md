# AD Gym

AD Gym could be discussed in several scenarios (to do), including:
- unsupervised
- semi-supervised
- supervised

### Data Augmentation
- Oversampling
- SMOTE
- Mixup
- GAN
- ...

### Data Preprocessing
- MinMax
- Normalization
- ...

### Network Architecture
- MLP
- AutoEncoder
- ResNet
- Transformer
- ...

### Network Initialization
- default (pytorch default)
- Xavier (uniform)
- Xavier (normal)
- Kaiming (uniform)
- Kaiming (normal)

### Network Hyperparameter
- neurons
- hidden layers
- batch normalization
- dropout
- activation layer
- ...

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

## Update Logs
- 2022.11.17: run the experiments of current component combinations
- 2022.11.23: add the GAN-based data augmentation method
- 2022.11.25: add the oversampling and SMOTE data augmentation method
- 2022.11.25: add the binary cross entropy loss and focal loss
- 2023.01.04: add the Mixup data augmentation method
- 2023.01.04: add different network initialization methods
- 2023.01.04: add the ordinal loss in PReNet model
- 2023.02.20: restart ADGym
- 2023.02.20: add two baselines: random selection and model selection based on the partially labeled data