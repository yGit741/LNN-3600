r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

student_name_1 = '' # string
student_ID_1 = '' # string
student_name_2 = '' # string
student_ID_2 = '' # string


# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
1. False: The test set allows us to estimate our out-of-sample error, not the in-sample error. In-sample error refers to the error rate or loss the model incurs 
over the training dataset on which it was trained. This measures how well the model fits the data it has seen. In contrast, the test set, which consists of unseen 
data, helps in estimating how well the model is likely to perform on new, unseen data (out-of-sample error). 
This is crucial for evaluating the model's generalization capability.

2. False: Not all splits of the data into two disjoint subsets would constitute equally useful train-test splits. 
The effectiveness of a train-test split depends on several factors:
Representativeness: The training set should be representative of the entire dataset to ensure that the model learns the underlying patterns applicable 
to the general population. Similarly, the test set should also be representative to ensure it can accurately reflect the model's performance in real-world scenarios.
Size: Typically, larger training sets allow the model to learn more comprehensive patterns, reducing the risk of underfitting. 
However, the test set also needs to be large enough to give a reliable estimate of the model's performance.
Randomness: If the split is done non-randomly (e.g., chronologically in time-series data), it might introduce bias. 
Random splits help in minimizing the bias and making the training and test data more representative.
Balance: In classification problems, it’s important that the proportion of each class is roughly the same in both training and testing datasets to avoid skewing 
the model performance.

3. True: The test set should indeed not be used during cross-validation. Cross-validation is a technique used to estimate the generalization performance of a model 
by dividing the training data into multiple subsets (folds). Each fold acts as a validation set in turn while the remaining folds are used for training. 
The purpose of this method is to validate the model's performance using different subsets of the data while still keeping the test set completely separate 
and untouched. This ensures that the test set remains a completely independent dataset for evaluating the model’s performance after the model has been fully trained 
and tuned, providing an unbiased assessment of its generalization capability.

4. True: The validation-set performance of each fold in cross-validation is indeed used as a proxy for the model’s generalization error. 
Cross-validation helps in reducing variability by using multiple different validation sets and averaging the performance across these sets. 
This averaged performance metric gives a more reliable estimate of how well the model is expected to perform on unseen data compared to using a single train-test split. 
This estimation is crucial for tuning model parameters and for choosing between different models or configurations.
"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
Your friend's approach of adding a regularization term to the loss function to address overfitting in a simple linear regression model is conceptually justified; 
however, the method used to select the regularization parameter λ has significant flaws.

Here’s a breakdown of the process and where it goes wrong:

Adding Regularization
Justified Reasoning: Overfitting occurs when a model learns noise or fluctuations specific to the training data, 
at the expense of failing to generalize to new, unseen data. 
In simple linear regression, although overfitting is less common due to the model's simplicity, it can still happen, 
especially if there are many predictors or collinear features. 
Regularization (like Ridge or Lasso for linear regression) adds a penalty on the size of the coefficients, which can effectively reduce overfitting by discouraging overly 
complex models. 
This part of the strategy is sound and a common practice.
Selecting λ Using Test Data is a flawed approach: The main issue lies in the method of selecting λ
Your friend used the test set to evaluate different values of λ and chose the one that gave the best performance on this test set. 
This approach is problematic for several reasons:
Misuse of Test Data: The test set should ideally be used only once, to evaluate the final model after all decisions about the model 
(including hyperparameters like λ) have been made. 
Using the test set to make decisions about the model (in this case, tuning λ) means that the test data is influencing the model development, 
which can lead to a biased estimate of model performance.
Leakage of Information: By repeatedly evaluating different models on the test set, information about the test set leaks into the model selection process. 
This can lead to overfitting on the test set itself, where the chosen λ is overly optimized for the test data rather than for general unseen data.
Correct Approach
Cross-Validation: Instead of using the test set for tuning λ, your friend should use techniques like k-fold cross-validation within the training set to choose 
λ. In k-fold cross-validation, the training set is split into k smaller sets (or folds), and the model is trained on k-1 of these folds, 
with the remaining part used as a mock test set (or validation set). 
This process is repeated k times, with each of the k folds used exactly once as the validation data. 
The results are then averaged to produce a more comprehensive evaluation of the model performance across various splits of the data.
Final Evaluation: Once λ is selected via cross-validation, the model can be re-trained on the entire training set using this chosen λ, 
and then finally evaluated on the separate, untouched test set. 
This gives an unbiased assessment of how well the model is likely to perform on genuinely new data.
In summary, while the intention behind using regularization was correct to address overfitting, 
the method of selecting λ by using the test set is not justified and could compromise the reliability of the model evaluation. 
Switching to a validation approach like cross-validation would be a more appropriate method to avoid bias and better estimate the model's generalization capability.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
