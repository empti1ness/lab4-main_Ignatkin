"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from grader_contracts.ensembles import EnsembleInput, EnsembleResult
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = getattr(data, "train_features", getattr(data, "X_train", None))
    y_train = getattr(data, "train_target", getattr(data, "y_train", None))
    X_test = getattr(data, "test_features", getattr(data, "X_test", None))
    y_test = getattr(data, "test_target", getattr(data, "y_test", None))

    if X_train is None or y_train is None or X_test is None or y_test is None:
        raise ValueError("Не удалось извлечь признаки и целевые переменные из EnsembleInput.")

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    model_name = "GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8)"
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, probabilities))
    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    try:
        return EnsembleResult(
            model_name=model_name,
            probabilities=probabilities,
            predictions=predictions,
            roc_auc=roc_auc,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
        )
    except TypeError:
        return EnsembleResult(
            model_name,
            probabilities,
            predictions,
            roc_auc,
            acc,
            prec,
            rec,
            f1,
        )