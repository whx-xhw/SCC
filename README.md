# Self-correcting Clustering
This is the source code of SelfCC.

## Abstract

The incorporation of target distribution significantly enhances the success of deep clustering. However, most of the related deep clustering methods suffer from two drawbacks: (1) manually-designed target distribution functions with uncertain performance and (2) cluster misassignment accumulation. To address these issues, a Self-correcting Clustering (Self-CC) framework is proposed. In Self-CC, a robust target distribution solver (RTDS) is designed to automatically predict the target distribution and alleviate the adverse influence of misassignments. Specifically, RTDS divides the high confidence samples selected according to the cluster assignments predicted by a clustering module into labeled samples with correct pseudo labels and unlabeled samples of possible misassignments by modeling its training loss distribution. With the divided data, RTDS can be trained in a semi-supervised way. The critical hyperparameter which controls the semi-supervised training process can be set adaptively by estimating the distribution property of misassignments in the pseudo-label space with the support of a theoretical analysis. The target distribution can be predicted by the well-trained RTDS automatically, optimizing the clustering module and correcting misassignments in the cluster assignments. The clustering module and RTDS mutually promote each other forming a positive feedback loop. Extensive experiments on four benchmark datasets demonstrate the effectiveness of the proposed Self-CC.


![Main Image](/img/fig.PNG)


## Download Pretrained Weights

You can download the pretrained model weights for different datasets from the following table:

| Dataset   | FE Link                                | Center Link                              | Backone Link                               |
|-----------|----------------------------------------|------------------------------------------|-----------------------------------------|
| USPS | [Download](https://drive.google.com/file/d/1yK9rzBEkHuhy2DAH-ioMRqgkqBYvUl5n/view?usp=sharing)   | [Download](https://drive.google.com/file/d/1lfpyChSy_XjndIxrobBhj5fccJhF59RV/view?usp=sharing) | -                                       |
| MNIST | [Download](https://drive.google.com/file/d/1VHNXn-Pv12sSpxrHbQUgukYyq7JdpKFC/view?usp=drive_link)   | [Download](https://drive.google.com/file/d/1-nskm52zKokX45_9JiXFnkh-ODlJE5ND/view?usp=drive_link) | -                                       |
| STL-10 | [Download](https://drive.google.com/file/d/1CEkzcuda1W7bt_U8iF7dOt8OSsEw3oia/view?usp=sharing)   | [Download](https://drive.google.com/file/d/14Z3OUcN8btiKLRFnnjwkPI6Wg_x-3gBo/view?usp=sharing) | [Download](https://drive.google.com/file/d/1DX8pNbptuaATjGyxzmco8B6TVgdJr1fP/view?usp=sharing) |
| CIFAR-10 | [Download](https://drive.google.com/file/d/1V7EaUc4UESXMQfzUDsbUxelFNmr3Y92X/view?usp=sharing)   | [Download](https://drive.google.com/file/d/1iFnUReqtINIcz93xF3o2pm64O8HpjrSS/view?usp=sharing) | [Download](https://drive.google.com/file/d/1pBvj8EIVItcoNJu3ohW1s9upOjx_Wj4_/view?usp=sharing) |

After download the pre-trained weights, these files are supposed to be located in './<dataset>/weight/'

For example, if you want to train a clustering model on USPS, these files should be saved to './usps/weight/' before training.
