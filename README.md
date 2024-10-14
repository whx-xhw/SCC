# Self-correcting Clustering

![Main Image](path/to/your/main-image.pdf)

## Abstract

The incorporation of target distribution significantly enhances the success of deep clustering. However, most of the related deep clustering methods suffer from two drawbacks: (1) manually-designed target distribution functions with uncertain performance and (2) cluster misassignment accumulation. To address these issues, a Self-correcting Clustering (Self-CC) framework is proposed. In Self-CC, a robust target distribution solver (RTDS) is designed to automatically predict the target distribution and alleviate the adverse influence of misassignments. Specifically, RTDS divides the high confidence samples selected according to the cluster assignments predicted by a clustering module into labeled samples with correct pseudo labels and unlabeled samples of possible misassignments by modeling its training loss distribution. With the divided data, RTDS can be trained in a semi-supervised way. The critical hyperparameter which controls the semi-supervised training process can be set adaptively by estimating the distribution property of misassignments in the pseudo-label space with the support of a theoretical analysis. The target distribution can be predicted by the well-trained RTDS automatically, optimizing the clustering module and correcting misassignments in the cluster assignments. The clustering module and RTDS mutually promote each other forming a positive feedback loop. Extensive experiments on four benchmark datasets demonstrate the effectiveness of the proposed Self-CC.

## Introduction

This repository contains the implementation of the method described in our paper "[Your Paper Title](link to your paper)" published in [Journal/Conference Name]. Our approach [Brief description of the method] achieves state-of-the-art results on [mention relevant tasks or datasets].

## Key Features

- [List key features of your method]
- [Highlight any unique aspects]
- [Performance summary or benchmarks]

## Installation

To use the code in this repository, follow the steps below:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the code as follows:

1. **Data Preparation**: 
   Place your dataset in the appropriate folder or download it as described in the [dataset documentation](link-to-dataset-documentation).

2. **Training**:
   To train the model, use the following command:
    ```bash
    python train.py --config config.yaml
    ```

3. **Evaluation**:
   After training, evaluate the model using:
    ```bash
    python evaluate.py --model path/to/saved/model
    ```

4. **Inference**:
   To use the trained model for inference on new data:
    ```bash
    python inference.py --input path/to/input --model path/to/saved/model
    ```

## Download Pretrained Weights

You can download the pretrained model weights for different datasets from the following table:

| Dataset   | FE Link                                | Center Link                              | Back Link                               |
|-----------|----------------------------------------|------------------------------------------|-----------------------------------------|
| Dataset 1 | [Download FE](https://example.com/dataset1/fe)   | [Download Center](https://example.com/dataset1/center) | -                                       |
| Dataset 2 | [Download FE](https://example.com/dataset2/fe)   | [Download Center](https://example.com/dataset2/center) | -                                       |
| Dataset 3 | [Download FE](https://example.com/dataset3/fe)   | [Download Center](https://example.com/dataset3/center) | [Download Back](https://example.com/dataset3/back) |
| Dataset 4 | [Download FE](https://example.com/dataset4/fe)   | [Download Center](https://example.com/dataset4/center) | [Download Back](https://example.com/dataset4/back) |

## Citation

If you use this code or dataset in your research, please cite our paper:

