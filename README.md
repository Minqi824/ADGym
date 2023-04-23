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

- For the experimental results of all the components, open the [test_ADGym.py](gym.py) and run:
```python
adgym = ADGym(la=5, grid_mode='small', grid_size=1000, gan_specific=False, suffix='test')
adgym.run()
```

- For the experimental results of all the current SOTA semi- or supervised models, open the [test_SOTA.py](sota.py) and run:
```python
pipeline = RunPipeline(suffix='SOTA', parallel='semi-supervise', mode='nla')
pipeline.run()

pipeline = RunPipeline(suffix='SOTA', parallel='supervise', mode='nla')
pipeline.run()
```

- For the experimental results of meta classifier (and its counterpart baseline), open the [meta.py](metaclassifier/meta_dl.py) and run:
```python
# two-stage meta classifier, using meta-feature extractor in MetaOD
run(suffix='', grid_mode='small', grid_size=1000, gan_specific=False, mode='two-stage')
# end-to-end meta classifier
run(suffix='', grid_mode='small', grid_size=1000, gan_specific=False, mode='end-to-end')
```

## Python Package Requirements
- iteration_utilities==0.11.0
- metaod==0.0.6
- scikit-learn==0.24
- imbalanced-learn==0.7.0
- torch==1.9.0
- tensorflow==2.8.0
- tabgan==1.2.1
- rtdl==0.0.13
- protobuf==3.20.*
- numpy==1.21.6

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
- 2023.02.22: provide both two-stage and end-to-end versions of meta predictor
- 2023.02.23: improve the training efficiency in meta classifier
- 2023.02.28: support GPU version of meta predictors and fix some bugs
- 2023.03.01: provide ml-based meta predictor
- 2023.03.01: using the performance rank ratio (instead of performance) as training targets
- 2023.04.23: learning-to-rank + ensemble strategy