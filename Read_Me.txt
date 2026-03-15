Machine Learning model to analyze design parameters and evaluate if it is safe, or not.
the data was generated synthetically, and the model was trained with dropout but evaluated with a baseline model without one for comparison.
Final model was testing with test data to evaluate the performance of the model.

# FEA Failure Prediction using Machine Learning

This project trains a Multi-Layer Perceptron (MLP) model to predict structural failure modes using synthetic engineering data.

## Inputs
- Max Stress
- Max Strain
- Displacement
- Temperature
- Young's Modulus
- Load Magnitude
- Thickness

## Outputs
- Safe
- Yield Failure
- Buckling
- Fatigue Risk

## Technologies
Python, PyTorch, Pandas, Scikit-learn