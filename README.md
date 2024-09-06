# **Deepfake Detection Using Dense CNN Architecture**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## **Project Overview**

Deepfakes, created using advanced deep learning techniques, pose significant challenges by generating highly convincing yet fake visual content. This project tackles the challenge of detecting such manipulated images using a Dense Convolutional Neural Network (CNN) architecture. Our model leverages EfficientNet, combined with a custom Depthwise Separable Convolution Block (DSCB), to enhance feature extraction while maintaining computational efficiency. The model has been evaluated on diverse datasets, achieving high accuracy in differentiating real from fake images.

### **Key Features**
- **Dense CNN with EfficientNet**: Uses a pre-trained EfficientNet model for efficient and accurate feature extraction.
- **Custom DSCB**: Introduces a Depthwise Separable Convolution Block for enhanced detection of deepfake artifacts.
- **High Accuracy**: Achieves 97% accuracy in detecting manipulated content.
- **Real-world Application**: Designed to combat misinformation, enhance media verification, and safeguard digital content integrity.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)

---

## **Installation**

### **Requirements**
- Python 3.x
- PyTorch
- Torchvision
- Timm (for pre-trained EfficientNet models)
- NumPy
- Pandas
- Matplotlib

Install the required packages:

```bash
# Clone the repository
git clone https://github.com/your_username/deepfake-detection.git

# Navigate to the project directory
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

---

## **Usage**

### **Training the Model**

To train the model on a dataset of deepfake images, use the following command:

```bash
python src/EfficientNet_Deepfake_Detection.py --train --epochs 20 --batch_size 32 --dataset ./datasets/training_data/
```

### **Testing the Model**

To test the model on a set of real and fake images, run:

```bash
python src/EfficientNet_Deepfake_Detection.py --test --dataset ./datasets/test_data/
```

---

## **Model Architecture**

### **Core Architecture**

The model combines the power of EfficientNet for general image classification with a custom Depthwise Separable Convolution Block (DSCB) designed to enhance the detection of deepfake artifacts. This allows the model to focus on both spatial details and inter-channel relationships in images, making it highly effective for deepfake detection.

```plaintext
Input Image
     |
Depthwise Separable Convolution Block (DSCB)
     |
EfficientNet (Pre-trained)
     |
Fully Connected Layer (Binary Classification: Real/Fake)
```

---

## **Results**

### **Performance Metrics**

- **Accuracy**: 97%
- **Precision**: 96%
- **Recall**: 98%
- **F1 Score**: 97%

The model was trained on a diverse dataset and demonstrated strong performance in distinguishing real and fake images across various scenarios. Below are sample plots illustrating training progress:

![Training Loss](./images/training_loss.png)
![Validation Accuracy](./images/validation_accuracy.png)

### **Example Results**

- **Real Image**: Correctly identified as real.
- **Deepfake Image**: Correctly identified as fake.

![Example Output](./images/example_output.png)

---

## **Contributing**

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes. Make sure to follow the [contribution guidelines](CONTRIBUTING.md).

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgments**

Special thanks to my collaborators and co-authors: Prem Kumar V, Abhishek Rithik O, and Chitra P from SRM Institute of Science and Technology.

---

By following this structure, your GitHub README will be clear, informative, and user-friendly, ensuring that visitors understand your project’s value and can easily get started with it.
