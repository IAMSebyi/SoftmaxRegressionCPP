# SoftmaxRegressionCPP

## Description

This project implements a Softmax Regression model from scratch in C++. Softmax Regression, also known as Multinomial Logistic Regression, is a generalization of Logistic Regression for multiclass classification. This implementation includes data normalization, model training using gradient descent, prediction, and evaluation functionalities.

### Features
- Data normalization using Z-score normalization
- Training with gradient descent
- Categorical Cross-Entropy loss calculation with L2 regularization
- Predicting class probabilities for new data points

## Formulas

### Softmax Function
The softmax function is used to convert the raw class scores (logits) into probabilities:

$$
\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

where:
- $\mathbf{z}$ is the vector of raw class scores
- $K$ is the number of classes

### Categorical Cross-Entropy Loss
The categorical cross-entropy loss measures the performance of the classification model:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$

where:
- $N$ is the number of data points
- $K$ is the number of classes
- $y_{i,k}$ is the true label (1 if the class is correct, 0 otherwise)
- $\hat{y}_{i,k}$ is the predicted probability for class $k$ for data point $i$

### L2 Regularization
L2 regularization is used to prevent overfitting by adding a penalty to the loss function:

$$
R = \frac{\lambda}{2} \sum_{j=1}^{K} \sum_{k=1}^{M} w_{j,k}^2
$$

where:
- $\lambda$ is the regularization parameter
- $K$ is the number of classes
- $M$ is the number of features
- $w_{j,k}$ is the weight for feature $k$ and class $j$

The final loss with L2 regularization is:

$$
L_{reg} = L + R
$$

### Gradient Descent Updates
The gradients of the loss function with respect to the weights and biases are used to update the parameters during training:

#### Gradient of the Loss with respect to the Weights

$$
\frac{\partial L}{ \partial w_{j,k} } = \frac{1}{N} \sum_{i=1}^{N} ( \hat{y}_ {i,j} - y_{i,j} ) x_{i,k} + \lambda w_{j,k}
$$

where:
- $w_{j,k}$ is the weight for feature $k$ and class $j$
- $\hat{y}_{i,j}$ is the predicted probability for class $j$ for data point $i$
- $y_{i,j}$ is the true label (1 if the class is correct, 0 otherwise)
- $x_{i,k}$ is the value of feature $k$ for data point $i$
- $\lambda$ is the regularization parameter

#### Gradient of the Loss with respect to the Biases

$$
\frac{\partial L}{\partial b_{j}} = \frac{1}{N} \sum_{i=1}^{N} ( \hat{y}_ {i,j} - y_{i,j} )
$$

where:
- $b_{j}$ is the bias for class $j$
- $\hat{y}_{i,j}$ is the predicted probability for class $j$ for data point $i$
- $y_{i,j}$ is the true label (1 if the class is correct, 0 otherwise)

### Weight and Bias Updates
During each iteration of gradient descent, the weights and biases are updated as follows:

$$
w_{j,k} = w_{j,k} - \eta \frac{\partial L}{\partial w_{j,k}}
$$

$$
b_{j} = b_{j} - \eta \frac{\partial L}{\partial b_{j}}
$$

where:
- $\eta$ is the learning rate
