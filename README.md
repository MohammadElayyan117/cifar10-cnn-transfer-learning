# CIFAR-10 Image Classification – CNN & Transfer Learning

A deep learning image-classification project using the CIFAR-10 dataset to compare a convolutional neural network trained from scratch with transfer learning based on a pretrained Xception model.

## Overview

The project trains and evaluates two image-classification approaches on CIFAR-10:

1. A custom CNN trained from scratch.
2. A transfer-learning model using pretrained Xception features.

The notebook includes data preparation, normalization, train/validation splitting, model training, test evaluation, and comparison plots for accuracy and loss.

## Dataset

CIFAR-10 contains 10 image classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The notebook uses:

- 37,500 training images
- 12,500 validation images
- 10,000 test images

Images are normalized and standardized before training, and labels are converted to one-hot encoded targets.

## Model 1 – CNN From Scratch

The custom CNN includes:

- `Conv2D`
- `ReLU`
- `MaxPooling2D`
- `Dropout`
- `Flatten`
- Dense layers
- Softmax classification output

Training configuration:

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch size: 64
- Epochs: 20

### Result

Test accuracy:

**~59%**

## Model 2 – Transfer Learning with Xception

The second model uses a pretrained Xception network with ImageNet weights.

CIFAR-10 images are upsampled from `32×32` to `224×224` before being passed into Xception.

The pretrained Xception convolutional layers are frozen, and a new classification head is added:

- `Flatten`
- Dense layer with 128 units
- 10-class softmax output

Training configuration:

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch size: 64
- Epochs: 5

### Result

Test accuracy:

**~77%**

The transfer-learning model substantially improves test accuracy compared with the CNN trained from scratch.

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| CNN from scratch | ~59% |
| Xception transfer learning | ~77% |

The notebook also plots training and validation accuracy and loss for both approaches.

## Repository File

- `cifar10_cnn_transfer_learning.ipynb` — complete notebook with preprocessing, training, evaluation, and model comparison
- `README.md` — project documentation

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- CNN
- Transfer Learning
- Xception
- Computer Vision
- Deep Learning

## Running the Notebook

The notebook can be run in:

- Google Colab
- Jupyter Notebook
- JupyterLab

Install the required packages if needed:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

Then open:

```text
cifar10_cnn_transfer_learning.ipynb
```

GPU execution is recommended for the Xception training stage.

## Author

**Mohammad Ahmad Elayyan**

- Email: [mohamadelayyan84@gmail.com](mailto:mohamadelayyan84@gmail.com)
- LinkedIn: https://www.linkedin.com/in/mohammadelayyan1
- GitHub: [MohammadElayyan117](https://github.com/MohammadElayyan117)
