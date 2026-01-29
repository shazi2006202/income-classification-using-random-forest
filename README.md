# income-classification-using-random-forest
A machine learning analysis using the Adult Income dataset to predict income levels. The project includes data cleaning, exploratory analysis, feature encoding, Random Forest classification, class imbalance handling using SMOTE, hyperparameter tuning, and model performance evaluation.
This repository contains a machine learning workflow applied to the Adult Income dataset, focusing on predicting whether an individual earns more than 50K per year based on demographic and employment-related attributes.

The process begins with data loading and preprocessing, where missing values are removed and the target variable is converted into a binary format. Categorical features such as workclass, education, occupation, and marital status are transformed into numerical values using label encoding to prepare the dataset for machine learning models.

Exploratory data analysis is carried out using visualizations to understand income distribution across education levels and age groups. Count plots and box plots are used to highlight patterns and differences between income categories.

A Random Forest classifier is trained using a stratified train-test split to preserve class distribution. Model performance is evaluated using accuracy, confusion matrices, and detailed classification reports. Feature importance analysis is conducted to identify the most influential factors contributing to income prediction.

To address class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) is applied, followed by hyperparameter tuning using RandomizedSearchCV. The tuned model is re-evaluated to measure improvements in predictive performance. Updated feature importance visualizations are generated to compare the drivers of high income before and after optimization.

Overall, the repository demonstrates a complete machine learning pipeline, including preprocessing, exploratory analysis, model training, imbalance handling, tuning, and evaluation using Python, Pandas, Scikit-learn, and visualization libraries
