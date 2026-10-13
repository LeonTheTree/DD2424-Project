# Comparing Semi-Supervised Learning Methods and Fine-Tuning Convolutional Networks

This repository contains the project code for Group 35 for the DD2424 Deep Learning course (2026). The project investigates the performance differences between various fine-tuning strategies and semi-supervised learning (SSL) techniques applied to an ImageNet pretrained ResNet-18 model. The core task is a multi-class classification problem recognizing 37 different breeds of cats and dogs using the Oxford-IIIT Pet Dataset. 

## Personal Contributions
Within the collaborative group, my primary responsibilities focused on the baseline classification models and the semi-supervised learning implementations and experiments:
* **Linear Probing & Baselines:** Implemented the binary and multi-class classification baselines by replacing the final layer of the pre-trained ConvNet and fine-tuning it using the Adam optimizer. Conducted the hyperparameter searches to establish optimal learning rates for these baselines.
* **Mean Teacher Implementation:** Built the Mean Teacher SSL pipeline utilizing consistency regularization, where the model is trained with unlabelled data using a student and teacher network. Implemented the exponential moving average (EMA) system to update the teacher's weights based on the student's weights.
* **SSL Evaluation:** Designed and executed the experiments comparing the SSL methods across varying fractions of labeled data (from 80% down to 1%). Our evaluations revealed that when fine-tuning a highly capable pre-trained model, using only the limited labeled data performs very similarly to the advanced SSL methods, ultimately achieving over 75% test accuracy with just 5% labeled data.

## 🛠 Requirements
The project is built using Python and requires the following libraries:
* `torch`
* `torchvision`
* `scikit-learn`
* `matplotlib`

## 🚀 How to Run
The main pipeline and experiments are contained within a single Jupyter Notebook. 

1. Clone the repository and open the notebook.
2. Run all cells in sequential order. 
3. **Note:** A GPU runtime is highly recommended due to the computationally intensive nature of the model fine-tuning. 

## 📊 Results
Key metrics and results are generated dynamically upon running the notebook. Because many of the original training iterations were executed locally, some final figures and detailed result tables are only visible once the notebook completes its execution in your environment.
