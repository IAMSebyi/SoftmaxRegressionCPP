#include "SoftmaxRegression.h"

// Constructor to initialize the LogisticRegression object with features, targets, and dimensions
SoftmaxRegression::SoftmaxRegression(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const int& numOfFeatures, const int& numOfDataPoints, const int& numOfClasses, const float& regularizationParam)
	: features(features), labels(labels), numOfFeatures(numOfFeatures), numOfDataPoints(numOfDataPoints), numOfClasses(numOfClasses), weights(numOfClasses, std::vector<float> (numOfFeatures, 0)), biases(numOfClasses, 0), regularizationParam(regularizationParam)
{
	Scale(); // Apply feature scaling upon initialization
}

// Scale the features using Z-Scale Normalization
void SoftmaxRegression::Scale()
{
	// Calculate mean and standard deviation of features
	mean = GetMean();
	standardDeviation = GetStandardDeviation(mean);

	// Apply Z-score normalization
	for (int i = 0; i < numOfDataPoints; i++) {
		for (int j = 0; j < numOfFeatures; j++) {
			if (standardDeviation[j] != 0) {
				features[i][j] = (features[i][j] - mean[j]) / standardDeviation[j];
			}
		}
	}
}

// Predict the probabilities of a given input vector to belong to all of the classes
std::vector<float> SoftmaxRegression::Predict(std::vector<float> input, const bool test) const
{
	if (test) {
		// Feature scaling of input test data
		for (int i = 0; i < numOfFeatures; i++) {
			input[i] = (input[i] - mean[i]) / standardDeviation[i];
		}
	}

	std::vector<float> probabilities(numOfClasses);

	for (int i = 0; i < numOfClasses; i++) {
		float denominator = 0;
		for (int j = 0; j < numOfClasses; j++) {
			// Calculate the denominator of the Softmax Function fraction using Linear Transformation
			denominator += std::exp(GetVectorMultiplication(weights[j], input) + biases[j]);
		}

		// Calculate probability using Softmax Function
		probabilities[i] = std::exp(GetVectorMultiplication(weights[i], input) + biases[i]) / denominator;
	}

	return probabilities;
}

// Calculate the Categorical Cross-Entropy loss with L2 regularization
float SoftmaxRegression::Loss() const
{
	float loss = 0;
	
	for (int i = 0; i < numOfDataPoints; i++) {
		std::vector<float> probabilities = Predict(features[i]);
		for (int j = 0; j < numOfClasses; j++) {
			// The ternary operator is used to One-Hot encode the true label
			loss += std::log(probabilities[j]) * ((labels[i] == j) ? 1 : 0);
		}
	}

	// Include L2 regularization in the loss calculation
	float regLoss = 0;
	for (const auto& weightVec : weights) {
		for (const auto& weight : weightVec) {
			regLoss += weight * weight;
		}
	}
	regLoss *= regularizationParam / 2.0f;

	loss = (loss / -numOfDataPoints) + regLoss;

	//loss /= -1 * numOfDataPoints;
	return loss;
}

// Calculate the gradient of the cost function with respect to weights matrix parameter
std::vector<std::vector<float>> SoftmaxRegression::WeightsGradient() const
{
	std::vector<std::vector<float>> weightsSlope(numOfClasses, std::vector<float>(numOfFeatures, 0));

	for (int i = 0; i < numOfDataPoints; i++) {
		std::vector<float> probabilities = Predict(features[i]);
		for (int j = 0; j < numOfClasses; j++) {
			for (int k = 0; k < numOfFeatures; k++) {
				weightsSlope[j][k] += (probabilities[j] - ((labels[i] == j) ? 1 : 0)) * features[i][k];
			}
		}
	}

	// Average the accumulated gradients and add regularization term
	for (int j = 0; j < numOfClasses; j++) {
		for (int k = 0; k < numOfFeatures; k++) {
			weightsSlope[j][k] /= numOfDataPoints;
			weightsSlope[j][k] += regularizationParam * weights[j][k];
		}
	}

	return weightsSlope;
}

// Calculate the gradient of the cost function with respect to the biases vector parameter
std::vector<float> SoftmaxRegression::BiasGradient() const
{
	std::vector<float> biasSlope(numOfClasses, 0);

	for (int i = 0; i < numOfDataPoints; i++) {
		std::vector<float> probabilities = Predict(features[i]);
		for (int j = 0; j < numOfClasses; j++) {
			biasSlope[j] += (probabilities[j] - ((labels[i] == j) ? 1 : 0));
		}
	}

	// Average the accumulated gradients
	for (int j = 0; j < numOfClasses; j++) {
		biasSlope[j] /= numOfDataPoints;
	}

	return biasSlope;
}

// Calculate the mean of the input features for feature scaling
std::vector<float> SoftmaxRegression::GetMean() const
{
	std::vector<float> mean(numOfFeatures, 0);

	for (int i = 0; i < numOfFeatures; i++) {
		for (int j = 0; j < numOfDataPoints; j++) {
			mean[i] += features[j][i];
		}
		mean[i] /= numOfDataPoints;
	}

	return mean;
}

// Calculate the standard deviation for feature scaling
std::vector<float> SoftmaxRegression::GetStandardDeviation(const std::vector<float>& mean) const
{
	std::vector<float> standardDeviation(numOfFeatures, 0);

	for (int i = 0; i < numOfFeatures; i++) {
		for (int j = 0; j < numOfDataPoints; j++) {
			standardDeviation[i] += (features[j][i] - mean[i]) * (features[j][i] - mean[i]);
		}
		standardDeviation[i] = std::sqrt(standardDeviation[i] / (numOfDataPoints - 1));
	}

	return standardDeviation;
}

// Calculate the multiplication of two vectors
float SoftmaxRegression::GetVectorMultiplication(const std::vector<float>& a, const std::vector<float>& b) const
{
	float result = 0;
	int i = 0;
	for (const auto& val : a) {
		result += val * b[i];
		i++;
	}
	return result;
}

// Train the model using Gradient Descent
void SoftmaxRegression::Train(const float& learningRate, const int maxIterations)
{
	const float convergenceThreshold = 2e-10; // Convergence threshold
	const int logStep = 1000; // Logging step interval

	int i;
	for (i = 1; i <= maxIterations; i++) {
		float prevLoss = Loss();
		std::vector<std::vector<float>> prevWeights = weights;
		std::vector<float> prevBiases = biases;

		std::vector<std::vector<float>> weightsSlope = WeightsGradient();
		std::vector<float> biasSlope = BiasGradient();

		for (int j = 0; j < numOfClasses; j++) {
			// Update weights matrix parameter
			for (int k = 0; k < numOfFeatures; k++) {
				weights[j][k] -= learningRate * weightsSlope[j][k];
			}
			// Update biases vector parameter
			biases[j] -= learningRate * biasSlope[j];
		}

		float currentLoss = Loss();

		// Check for convergence
		if (i > 1 && (currentLoss > prevLoss || prevLoss - currentLoss <= convergenceThreshold)) {
			// Early stoppage due to convergence
			weights = prevWeights;
			biases = prevBiases;
			break;
		}

		if (i % logStep == 0) {
			std::cout << "Iteration #" << i << ", Cost: " << currentLoss << '\n';
		}
	}

	// Output final cost to console for learning rate adjustments
	std::cout << "Final cost after " << i - 1 << " iterations : " << Loss() << "\n \n";
}

// Get the model parameters (coefficients and intercept)
std::vector<std::vector<float>> SoftmaxRegression::GetParameters() const
{
	std::vector<std::vector<float>> parameters (weights);
	parameters.push_back(biases);
	return parameters;
}