# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This project ofuses on building a binary classification model to predict whether a banking client will subscribe to a term deposit based on demographic and socioeconomic attribites from the UCI Bank Marketing dataset.

Solution: Multiple modes were trained using AzureML HyperDrive and Microsoft AutoML. The best HyperDrive Logistic Regression model achoeved approximately 91% accuracy, while AutoML's VotingEnsemble model delivered slightly better performance overall.

## Scikit-learn Pipeline

**Pipeline Architecture:**
The pipeline loads and preprocesses the Bank Marketing dataset, followed by trainig a Logistic Regression classifier using scikit-learn. Hyperparameter tuning is performed with AzureML HyperDrive, where multiple model configurations are evaluated in parallel to identify the best-performing model based on accuracy.

**Parameter Sampler Benefits:**
Random parameter sampling was used, allowing efficient exploration of the hyperparameter search space without evaluating every possible combination. This approach reduces computational cost while still providing a high likelihood of discovering optimar or near-optimal parameter values.


**Early Stopping Policy Benefits:**
A Bandit early stopping policy was applied to terminate underperforming runs early. This provides resource efficiency by focusing computation on promising configurations and reducing unnecessary trainig time.

## AutoML
**AutoML Model and Hypeparameters:**
Azure AutoML automatically trained and evaluated multiple classification algorithms with different preprocessing steps and hyperparameter settings. The best-performing model was VotingEnsemble, which combined predictions from several individual models to achieve improved overall performance.

## Pipeline comparison
The HyperDrive approach using Logistic Regression achieved strong performace with an accuracy of approximately 91%. AutoML produced a VotingEnsemble model that slightly outperformed the best HyperDrive model. The difference in accuracy can be attributed to AutoML's ability to evaluate multiple algorithms, preprocessing techniques, and hyperparameter combinations, whereas the HyperDrive pipeline was limited to a single classification algorithm. Architectually, AutoML leverages model ensembling, which improved generalization by combining predictions from several models.

## Future work
Future experiments could explore a broader range of algorithms with HyperDriver, such as tree-based or boosting models to improve prediction performance. Additional feature engineering, such a handling class imbalance or creating interantion features, could further enhance model accuracy. Increasing the search space for hyperparameters or using cross-validation may also lead to more robust and generalizable models.

## Proof of cluster clean up
The computer cluster used for trainig was deleted after completing the experiments to avoid unnecessary resource usage and costs.
**Image of cluster marked for deletion**
