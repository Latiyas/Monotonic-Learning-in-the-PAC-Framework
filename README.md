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

# Experiment on Conjunction of Boolean literal conjunction problem:
python BooleanConjunctions.py --num_repeat 1000 --num_span 100 --num_mini_sample 25 --num_iter 50

# Experiment on threshold function problem:
python ThresholdFunction.py --num_repeat 1000 --num_span 100 --num_mini_sample 200 --num_iter 50

# Experiment on Iris classification problem:
python Iris.py --num_repeat 1000 --num_span 100 --num_mini_sample 200 --num_iter 50

# Experiment on Boolean literal conjunction problem with small sample size:
python BooleanConjunctions.py --num_repeat 100 --num_span 100 --num_mini_sample 5 --num_iter 50

# Analysis of P_m and Q_m with different sample sizes (Figures 3-8, D.10, and D.11 in the paper):
python DataAnalysis.py --data_type CBL --num_repeat 1000 --num_span 100 --num_mini_sample 25 --num_iter 50
python DataAnalysis.py --data_type TH --num_repeat 1000 --num_span 100 --num_mini_sample 200 --num_iter 50
python DataAnalysis.py --data_type Iris --num_repeat 1000 --num_span 100 --num_mini_sample 200 --num_iter 50
python DataAnalysis.py --data_type CBL --num_repeat 100 --num_span 100 --num_mini_sample 5 --num_iter 50 --extra_name samll

# Analyze the monotonicity of the empirical distribution (consider the mean value for the Boolean literal conjunction problem):
python AnalysisMonotone.py --num_repeat 1000 --num_span 100 --num_mini_sample 25 --num_iter 50
python AnalysisMonotone.py --num_repeat 100 --num_span 100 --num_mini_sample 5 --num_iter 50

# Analyze the influence of different theoretical bounds (Figure 9 in the paper):
python CompareTheoreticalBound.py
```