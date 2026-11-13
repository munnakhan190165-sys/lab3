from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.array(data.X_train, dtype=float)
    y_train = np.array(data.y_train, dtype=float)
    X_test = np.array(data.X_test, dtype=float)
    
    # Add bias term (column of 1s)
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # Gradient Descent Hyperparameters
    learning_rate = 0.01
    epochs = 1000
    m = len(y_train)
    
    weights = np.zeros(X_train_b.shape[1])
    mse_history = []
    
    for _ in range(epochs):
        predictions = X_train_b.dot(weights)
        error = predictions - y_train
        mse = np.mean(error ** 2)
        mse_history.append(float(mse))
        
        gradient = (2 / m) * X_train_b.T.dot(error)
        weights = weights - learning_rate * gradient
        
    intercept = float(weights[0])
    coef = weights[1:].tolist()
    test_predictions = X_test_b.dot(weights).tolist()
    
    return RegressionResult(
        coefficients=coef,
        intercept=intercept,
        mse_history=mse_history,
        predictions=test_predictions
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.array(data.X_train, dtype=float)
    y_train = np.array(data.y_train, dtype=float)
    X_test = np.array(data.X_test, dtype=float)
    
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    learning_rate = 0.1
    epochs = 1000
    m = len(y_train)
    
    weights = np.zeros(X_train_b.shape[1])
    loss_history = []
    
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    for _ in range(epochs):
        z = X_train_b.dot(weights)
        y_pred = sigmoid(z)
        
        # Binary Cross-Entropy Loss
        loss = -np.mean(y_train * np.log(y_pred + 1e-15) + (1 - y_train) * np.log(1 - y_pred + 1e-15))
        loss_history.append(float(loss))
        
        gradient = (1 / m) * X_train_b.T.dot(y_pred - y_train)
        weights = weights - learning_rate * gradient

    intercept = float(weights[0])
    coef = weights[1:].tolist()
    
    test_probs = sigmoid(X_test_b.dot(weights))
    test_preds_binary = (test_probs >= 0.5).astype(int).tolist()
    
    return ClassificationResult(
        coefficients=coef,
        intercept=intercept,
        loss_history=loss_history,
        probabilities=test_probs.tolist(),
        predictions=test_preds_binary
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.array(data.X_train, dtype=float)
    y_train = np.array(data.y_train, dtype=float)
    X_test = np.array(data.X_test, dtype=float)
    y_test = np.array(data.y_test, dtype=float)
    
    # 1. Own Logistic Regression
    custom_res = train_logistic_classifier(data)
    custom_preds = custom_res.predictions
    
    own_logistic_metrics = {
        "accuracy": float(accuracy_score(y_test, custom_preds)),
        "precision": float(precision_score(y_test, custom_preds, zero_division=0)),
        "recall": float(recall_score(y_test, custom_preds, zero_division=0)),
        "f1": float(f1_score(y_test, custom_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, custom_res.probabilities))
    }
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_metrics = {
        "accuracy": float(accuracy_score(y_test, dt_preds)),
        "precision": float(precision_score(y_test, dt_preds, zero_division=0)),
        "recall": float(recall_score(y_test, dt_preds, zero_division=0)),
        "f1": float(f1_score(y_test, dt_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]))
    }
    
    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_metrics = {
        "accuracy": float(accuracy_score(y_test, knn_preds)),
        "precision": float(precision_score(y_test, knn_preds, zero_division=0)),
        "recall": float(recall_score(y_test, knn_preds, zero_division=0)),
        "f1": float(f1_score(y_test, knn_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1]))
    }
    
    return BaselineComparisonResult(
        own_logistic=own_logistic_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )

