"""
classifier.py — Random Forest classifier wrapper for particle track
classification.

Provides functions to create, train, evaluate, save, and load the model,
and to predict particle types with probability estimates.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import config


def create_model():
    """
    Create a new Random Forest classifier with the hyperparameters
    defined in config.py.

    Returns
    -------
    model : RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )
    return model


def train_model(X, y, cv_folds=5):
    """
    Train a Random Forest classifier and evaluate with cross-validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    model : RandomForestClassifier
        The trained model (fitted on ALL data).
    cv_accuracy : float
        Mean cross-validation accuracy.
    cv_std : float
        Standard deviation of cross-validation accuracy.
    report : str
        Classification report from cross-validation predictions.
    """
    model = create_model()

    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                          random_state=config.RF_RANDOM_STATE)

    # Only do CV if we have enough samples per class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()

    if min_class_count >= cv_folds:
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        # Too few samples for stratified CV; use leave-one-out or simple split
        from sklearn.model_selection import LeaveOneOut
        cv_scores = cross_val_score(
            model, X, y,
            cv=min(min_class_count, cv_folds),
            scoring="accuracy",
        )
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()

    # Train on full dataset
    model.fit(X, y)

    # Generate classification report on training data (for reference)
    y_pred = model.predict(X)
    label_names = [config.PARTICLE_LABELS[i] for i in sorted(unique_classes)]
    report = classification_report(
        y, y_pred, target_names=label_names, zero_division=0
    )

    return model, cv_accuracy, cv_std, report


def predict_track(model, features):
    """
    Predict the particle type for a single track.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model.
    features : np.ndarray, shape (12,)
        Feature vector for a single track.

    Returns
    -------
    predicted_class : int
        Predicted class index.
    probabilities : dict
        {class_index: probability} for all classes.
    """
    features_2d = features.reshape(1, -1)
    predicted_class = int(model.predict(features_2d)[0])
    proba = model.predict_proba(features_2d)[0]

    probabilities = {}
    for i, cls in enumerate(model.classes_):
        probabilities[int(cls)] = float(proba[i])

    return predicted_class, probabilities


def get_confusion_matrix(model, X, y):
    """
    Compute the confusion matrix for the training data.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix.
    labels : list of str
        Class label names.
    """
    y_pred = model.predict(X)
    unique_classes = sorted(np.unique(np.concatenate([y, y_pred])))
    labels = [config.PARTICLE_LABELS[int(c)] for c in unique_classes]
    cm = confusion_matrix(y, y_pred, labels=unique_classes)
    return cm, labels


def save_model(model, path=None):
    """Save the trained model to disk."""
    if path is None:
        path = config.MODEL_PATH
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=None):
    """Load a trained model from disk."""
    if path is None:
        path = config.MODEL_PATH
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
