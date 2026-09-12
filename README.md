# Student Performance Prediction

## Project Overview

This project focuses on predicting secondary school students final academic grades (G3, on a scale of 0 to 20) using demographic information, family background, social habits, academic history, and previous period marks (G1 and G2).

## Dataset Link

https://www.kaggle.com/datasets/uciml/student-alcohol-consumption

## Approach

1. Exploratory Data Analysis
   Loaded the student dataset containing 649 records and 33 attributes.
   Verified data cleanliness and confirmed zero missing values across all columns.
   Analyzed individual variable distributions including student age, urban or rural residence, reason for school choice, and alcohol intake.
   Investigated bivariate and multivariate relationships to understand how study duration, parental education, absences, and internet access influence academic performance.
   Analyzed numerical distributions and spread using interquartile range techniques.

2. Data Preprocessing and Pipeline Creation
   Separated features into 17 categorical attributes and 15 numerical attributes.
   Applied OneHotEncoder to categorical features with handle unknown set to ignore.
   Configured a ColumnTransformer to process categorical and numerical features cleanly without data leakage.
   Divided data into a 75% training set (486 records) and a 25% test set (163 records) with a fixed random seed of 42.

3. Model Training and Benchmarking
   Built scikit learn pipelines for six regression models:
   Linear Regression
   Ridge Regression
   Lasso Regression
   Decision Tree Regressor
   Random Forest Regressor
   Gradient Boosting Regressor
   Evaluated each model on unseen test data using Mean Absolute Error (MAE), Mean Squared Error (MSE), R squared score (R2), and accuracy within a 1 mark tolerance.

4. Hyperparameter Optimization
   Applied grid search cross validation on Lasso Regression.
   Conducted coarse tuning followed by fine tuning to determine the optimal regularization parameter.
   Determined the optimal alpha value to be 0.19.

5. Web Application Development
   Saved the trained Lasso pipeline and dataset metadata.
   Created an interactive Streamlit application in app.py that allows users to input student attributes and receive instant grade predictions, performance category classifications, and personalized guidance notes.

## Key Insights

1. Past Academic Performance
   First period grade (G1) and second period grade (G2) have the strongest positive correlation with the final grade (G3). Prior academic performance is the single most reliable predictor of future success.

2. Weekly Study Time
   Students who dedicate 5 to 10 hours per week to studying achieve higher average grades than students who study less.

3. Internet Access
   Students with home internet access demonstrate better average academic outcomes compared to those without.

4. Absences and Prior Failures
   Excessive school absences and past class failures have a strong negative association with final grades.

5. Alcohol Consumption
   Higher workday and weekend alcohol consumption correlates with lower academic performance.

6. Parental Education
   Higher levels of education for both mothers and fathers show a positive relationship with student grades.

7. Residential Area
   Students living in urban areas tend to obtain slightly higher average scores than students living in rural areas in this dataset.

## Model Benchmark Comparison

Evaluation results on the 25% unseen test set (163 samples):

```
Model Benchmark Comparison

Rank  Model                MAE     MSE     R2 Score  Accuracy
1     Lasso Regression     0.6773  1.1869  0.8823    80.98%
2     Ridge Regression     0.7737  1.3565  0.8654    70.55%
3     Linear Regression    0.7750  1.3583  0.8653    70.55%
4     Random Forest        0.7759  1.4981  0.8514    74.23%
5     Gradient Boosting    0.7744  1.6175  0.8395    73.62%
6     Decision Tree        0.9264  2.9141  0.7109    84.66%
```

## Best Model

The best model for this project is Lasso Regression.

Reasons for selecting Lasso Regression:

1. Highest Predictive Accuracy: Achieved the highest R squared score of 0.8823 on the baseline benchmark and 0.8806 when fine tuned.
2. Lowest Error: Achieved the lowest Mean Absolute Error (0.6773 baseline, 0.7189 tuned) and lowest Mean Squared Error (1.1869 baseline, 1.2034 tuned).
3. Practical Precision: Yielded 82.21% accuracy within a 1 mark tolerance range on the test dataset after tuning.
4. L1 Regularization Benefit: Lasso performs automatic feature selection by shrinking coefficients of less informative features to zero, preventing overfitting and ensuring strong generalization.

## How to Run

1. Run the Streamlit web application:
   streamlit run app.py

2. Open the Jupyter Notebook for complete analysis and modeling:
   Student_Performance.ipynb
