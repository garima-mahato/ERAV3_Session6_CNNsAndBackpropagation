 # ML Model CI/CD Pipeline

 [![ML Model Tests](https://github.com/garima-mahato/ERAV3_Session6_CNNsAndBackpropagation/actions/workflows/ml-tests.yml/badge.svg)](https://github.com/garima-mahato/ERAV3_Session6_CNNsAndBackpropagation/actions/workflows/ml-tests.yml)


| Jupyter Notebook Link | https://github.com/garima-mahato/ERAV3_Session6_CNNsAndBackpropagation/blob/main/ERA_V3_Session6.ipynb |
|---|---|


## Model Architecture
- CNN with below architecture:
```
├── Input Block
│ ├── 3x3 Conv2D (16 channels)
│ ├── ReLU
│ ├── BatchNorm2d
│ └── Dropout
├── Convolutional Block 1
│ ├── 3x3 Conv2D (24 channels)
│ ├── ReLU
│ ├── BatchNorm2d
│ └── Dropout
├── Transition Block 1
│ ├── 1x1 Conv2D (10 channels)
│ └── MaxPool2d
├── Convolutional Block 2
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (14 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (16 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (16 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
├── Output Block
│ ├── Global Average Pooling (6 channels)
│ └── 1x1 Conv2D (16 outputs)
│ └── 1x1 Conv2D (32 outputs)
│ └── Fully Connected Layer (10 outputs)
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 24, 24, 24]           3,456
              ReLU-6           [-1, 24, 24, 24]               0
       BatchNorm2d-7           [-1, 24, 24, 24]              48
           Dropout-8           [-1, 24, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             240
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 14, 10, 10]           1,260
             ReLU-12           [-1, 14, 10, 10]               0
      BatchNorm2d-13           [-1, 14, 10, 10]              28
          Dropout-14           [-1, 14, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,016
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
        AvgPool2d-23             [-1, 16, 1, 1]               0
           Conv2d-24             [-1, 16, 1, 1]             256
           Conv2d-25             [-1, 32, 1, 1]             512
           Linear-26                   [-1, 10]             330
================================================================
Total params: 10,690
Trainable params: 10,690
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.90
Params size (MB): 0.04
Estimated Total Size (MB): 0.94
----------------------------------------------------------------
```

- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Total parameters:  10,690
- Hyperparameters:
    - Epochs: 20
    - Learning Rate: 0.15
    - Batch Size: 64
    - Dropout: 0.05
    - Optimizer: SGD
    - Loss Function: NLLLoss
    - Device: GPU
    - Dataset: MNIST
    - LR Scheduler: StepLR
    - SEED: 1
    - Image Size: 28x28x1
    - Image Augmentation: Rotation, Scaling, Translation


## Training Logs:
```
Epoch 1
Train: Loss=0.9541 Batch_id=937 Accuracy=86.61: 100%|██████████| 938/938 [00:35<00:00, 26.72it/s]
Test set: Average loss: 0.0890, Accuracy: 9728/10000 (97.28%)

Epoch 2
Train: Loss=0.0378 Batch_id=937 Accuracy=94.84: 100%|██████████| 938/938 [00:35<00:00, 26.68it/s]
Test set: Average loss: 0.0539, Accuracy: 9829/10000 (98.29%)

Epoch 3
Train: Loss=0.2178 Batch_id=937 Accuracy=95.76: 100%|██████████| 938/938 [00:35<00:00, 26.78it/s]
Test set: Average loss: 0.0618, Accuracy: 9808/10000 (98.08%)

Epoch 4
Train: Loss=0.0882 Batch_id=937 Accuracy=96.01: 100%|██████████| 938/938 [00:34<00:00, 27.13it/s]
Test set: Average loss: 0.0487, Accuracy: 9858/10000 (98.58%)

Epoch 5
Train: Loss=0.0537 Batch_id=937 Accuracy=96.35: 100%|██████████| 938/938 [00:33<00:00, 27.87it/s]
Test set: Average loss: 0.0684, Accuracy: 9790/10000 (97.90%)

Epoch 6
Train: Loss=0.3691 Batch_id=937 Accuracy=96.81: 100%|██████████| 938/938 [00:33<00:00, 27.87it/s]
Test set: Average loss: 0.0799, Accuracy: 9784/10000 (97.84%)

Epoch 7
Train: Loss=0.3405 Batch_id=937 Accuracy=96.69: 100%|██████████| 938/938 [00:33<00:00, 28.01it/s]
Test set: Average loss: 0.0457, Accuracy: 9867/10000 (98.67%)

Epoch 8
Train: Loss=0.1556 Batch_id=937 Accuracy=97.01: 100%|██████████| 938/938 [00:34<00:00, 27.21it/s]
Test set: Average loss: 0.0324, Accuracy: 9908/10000 (99.08%)

Epoch 9
Train: Loss=0.0146 Batch_id=937 Accuracy=97.13: 100%|██████████| 938/938 [00:34<00:00, 27.43it/s]
Test set: Average loss: 0.0327, Accuracy: 9904/10000 (99.04%)

Epoch 10
Train: Loss=0.0865 Batch_id=937 Accuracy=97.16: 100%|██████████| 938/938 [00:34<00:00, 27.24it/s]
Test set: Average loss: 0.0317, Accuracy: 9906/10000 (99.06%)

Epoch 11
Train: Loss=0.0886 Batch_id=937 Accuracy=97.31: 100%|██████████| 938/938 [00:34<00:00, 27.52it/s]
Test set: Average loss: 0.0390, Accuracy: 9880/10000 (98.80%)

Epoch 12
Train: Loss=0.1826 Batch_id=937 Accuracy=97.25: 100%|██████████| 938/938 [00:33<00:00, 27.80it/s]
Test set: Average loss: 0.0358, Accuracy: 9884/10000 (98.84%)

Epoch 13
Train: Loss=0.0432 Batch_id=937 Accuracy=98.36: 100%|██████████| 938/938 [00:33<00:00, 28.01it/s]
Test set: Average loss: 0.0182, Accuracy: 9935/10000 (99.35%)

Epoch 14
Train: Loss=0.0311 Batch_id=937 Accuracy=98.47: 100%|██████████| 938/938 [00:34<00:00, 27.25it/s]
Test set: Average loss: 0.0167, Accuracy: 9946/10000 (99.46%)

Epoch 15
Train: Loss=0.0016 Batch_id=937 Accuracy=98.56: 100%|██████████| 938/938 [00:34<00:00, 27.28it/s]
Test set: Average loss: 0.0167, Accuracy: 9945/10000 (99.45%)

Epoch 16
Train: Loss=0.0128 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:34<00:00, 27.31it/s]
Test set: Average loss: 0.0170, Accuracy: 9949/10000 (99.49%)

Epoch 17
Train: Loss=0.0037 Batch_id=937 Accuracy=98.55: 100%|██████████| 938/938 [00:33<00:00, 28.24it/s]
Test set: Average loss: 0.0174, Accuracy: 9949/10000 (99.49%)

Epoch 18
Train: Loss=0.0005 Batch_id=937 Accuracy=98.60: 100%|██████████| 938/938 [00:33<00:00, 28.35it/s]
Test set: Average loss: 0.0179, Accuracy: 9943/10000 (99.43%)

Epoch 19
Train: Loss=0.0338 Batch_id=937 Accuracy=98.55: 100%|██████████| 938/938 [00:33<00:00, 27.92it/s]
Test set: Average loss: 0.0158, Accuracy: 9949/10000 (99.49%)

Epoch 20
Train: Loss=0.0924 Batch_id=937 Accuracy=98.54: 100%|██████████| 938/938 [00:34<00:00, 27.33it/s]
Test set: Average loss: 0.0159, Accuracy: 9949/10000 (99.49%)
```

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest

## Local Setup

###1. Clone the repository
```bash
git clone <repository-url>
cd <folder-path>
```

### 2. Create and activate virtual environment
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
``` 

###4. Run tests
```
cd src
python -m pytest test_model.py -v
```


## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:
1. Sets up a Python environment
2. Installs dependencies
3. Runs all tests

### [Tests Include](https://github.com/garima-mahato/ERAV3_Session6_CNNsAndBackpropagation/blob/main/tests/test_model.py):
- Model parameter count verification (< 20,000 parameters)
- Check if Batch Normalization is used
- Check if Dropout is used
- Check if Global Average Pooling is used

## File Descriptions

- `src/model.py`: Contains the CNN architecture
- `tests/test_model.py`: Test suite for model validation
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow configuration
- `requirements.txt`: Project dependencies


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.