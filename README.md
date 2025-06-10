# Probably Approximately Monotone
Code Appendix for Probably Approximately Monotone

## Requirements
- Python 3.12.7

## Installation
Install numpy, matplotlib, scipy, sklearn from the web.


## Document organization structure
```
|-- code                			# code
|-- data                			# data set
|-- prediction_result				# final result will be stored here
|-- user_data           			# user data
	|-- model_data        			# to save model file
	|-- tmp_data        			# to save temporary file
```

## Usage
```
# Enter code path:
cd code

# Obtain a theoretical distribution (Figures 1 and 2 in the paper):
python TheoreticalDistribution.py

# Experiment on Boolean connection problem:
python BooleanConjunctions.py

# Experiment on threshold function problem:
python ThresholdFunction.py

# Experiment on threshold function problem:
python Iris.py

# Analysis of P_m and Q_m with different sample sizes (Figures 3 and 8 in the paper):
python DataAnalysis.py

# Analyze the monotonicity of the empirical distribution (consider the mean value for the Boolean literal conjunction problem):
python AnalysisMonotone.py

# Analyze the influence of different theoretical bounds (Figure 9 in the paper):
python CompareTheoreticalBound.py
```