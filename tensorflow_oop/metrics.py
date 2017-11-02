"""
Metrics without TensorFlow.
"""

import numpy as np


def r2_score(y_true, y_pred, sample_weight=None):
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    assert len(y_true) == len(y_pred)
    if sample_weight is not None:
        assert sample_weight.ndim == 1
        assert len(sample_weight) == len(y_pred)
    
    if sample_weight is not None:
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    y_average = np.average(y_true, axis=0, weights=sample_weight)
    denominator = (weight * (y_true - y_average) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])

    # Arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    avg_weights = denominator

    # Avoid fail on constant y or one-element arrays
    if not np.any(nonzero_denominator):
        if not np.any(nonzero_numerator):
            return 1.0
        else:
            return 0.0

    return np.average(output_scores, weights=avg_weights)

def accuracy_score(y_true, y_pred, sample_weight=None):
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    assert len(y_true) == len(y_pred)
    if sample_weight is not None:
        assert sample_weight.ndim == 1
        assert len(sample_weight) == len(y_pred)

    score = y_true == y_pred

    return np.average(score, weight=sample_weight)
