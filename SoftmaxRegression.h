#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class SoftmaxRegression
{
public:
	// Constructor
	SoftmaxRegression(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, const int& numOfFeatures, const int& numOfDataPoints, const int& numOfClasses, const float& regularizationParam);

	// Public methods
	std::vector<float> Predict(std::vector<float> input, const bool test = false) const;
	void Train(const float& learningRate, const int maxIterations = 40000);
	std::vector<std::vector<float>> GetParameters() const;

private:
	// Private methods
	void Scale();
	float Loss() const;
	std::vector<std::vector<float>> WeightsGradient() const;
	std::vector<float> BiasGradient() const;
	std::vector<float> GetMean() const;
	std::vector<float> GetStandardDeviation(const std::vector<float>& mean) const;
	float GetVectorMultiplication(const std::vector<float>& a, const std::vector<float>& b) const;

	// Data members
	std::vector<std::vector<float>> features;
	std::vector<int> labels;
	int numOfClasses;

	// Model parameters
	std::vector<std::vector<float>> weights;
	std::vector<float> biases;

	// Variables
	int numOfFeatures;
	int numOfDataPoints;
	float regularizationParam;
	std::vector<float> mean;
	std::vector<float> standardDeviation;
};
