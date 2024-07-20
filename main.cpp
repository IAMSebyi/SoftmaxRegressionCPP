#include "SoftmaxRegression.h"
#include <fstream>
#include <cstring>
#include <map>
#include <algorithm>

int main() {
	// Open the input and output files
	std::ifstream inputData("data.txt");
	std::ifstream inputTest("test.txt");
	std::ofstream outputParameters("parameters.txt");

	int numOfFeatures, numOfDataPoints, maxIterations = -1;
	float learningRate, regularizationParam = 0;

	// Read number of features and data points
	inputData >> numOfFeatures >> numOfDataPoints;

	// Initialize vectors for features and labels
	std::vector<std::vector<float>> features(numOfDataPoints, std::vector<float>(numOfFeatures)); // Input independent variables
	std::vector<std::string> labels(numOfDataPoints); // Output dependent variables

	std::vector<int> encodedLabels(numOfDataPoints);
	std::map<int, std::string> codebook;
	int numOfClasses = 0;

	// Read features and targets from input file
	for (int i = 0; i < numOfDataPoints; i++) {
		for (int j = 0; j < numOfFeatures; j++) {
			inputData >> features[i][j];
		}
		inputData >> labels[i];

		// Integer encode label
		bool newClass = true;
		for (int j = 0; j < numOfClasses; j++) {
			// Check if label is already encoded
			if (labels[i] == codebook[j]) {
				encodedLabels[i] = j;
				newClass = false;
			}
		}
		// Insert new mapping
		if (newClass) {
			codebook[numOfClasses] = labels[i];
			encodedLabels[i] = numOfClasses;
			numOfClasses++;
		}
	}

	// Read learning rate and max iterations
	inputData >> learningRate >> maxIterations >> regularizationParam;

	// Close data input file stream
	inputData.close();

	// Create Softmax Regression model
	SoftmaxRegression model(features, encodedLabels, numOfFeatures, numOfDataPoints, numOfClasses, regularizationParam);

	// Train the model
	if (maxIterations == -1) model.Train(learningRate);
	else model.Train(learningRate, maxIterations);

	// Get the model parameters (coefficients and intercept)
	std::vector<std::vector<float>> parameters;
	parameters = model.GetParameters();

	// Write the model parameters to output file
	for (const std::vector<float>& paramVec : parameters) {
		for (const float& param : paramVec) {
			outputParameters << param << " ";
		}
		outputParameters << '\n';
	}

	// Close parameters output file stream
	outputParameters.close();

	// Check if test data points are provided
	int numOfTestDataPoints = -1;
	inputTest >> numOfTestDataPoints;

	// Predict and print results for test data points
	if (numOfTestDataPoints != -1) {
		std::vector<float> testFeature(numOfFeatures);

		for (int i = 0; i < numOfTestDataPoints; i++) {
			std::cout << "Result for input { ";

			for (int j = 0; j < numOfFeatures; j++) {
				inputTest >> testFeature[j];
				std::cout << testFeature[j] << " ";
			}

			std::vector<float> probability = model.Predict(testFeature, true);
			std::cout << "}: " << codebook[std::distance(probability.begin(), std::max_element(probability.begin(), probability.end()))] << '\n';
		}
	}

	// Close test input file stream
	inputTest.close();

	return 0;
}