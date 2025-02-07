import numpy as np


def teca_score(Y_true, Y_pred):
    P = Y_true.sum(1)
    teca = (1 - abs(Y_true - Y_pred).sum(1) / 2 / P).mean()

    return teca


def f1_score(Y_test, Y_pred, average='samples', weights=None):
    assert Y_test.shape == Y_pred.shape

    # Calculate error
    TP = (Y_pred > 0) & (Y_test > 0)
    FP = (Y_pred > 0) & (Y_test == 0)
    FN = (Y_pred <= 0) & (Y_test > 0)

    if average == 'micro':
        axis = None
    elif average == 'macro':
        axis = 0
    elif average == 'samples':
        axis = 1
    elif average == 'weighted':
        axis = 0
        assert weights is not None
        assert len(weights) == Y_test.shape[1]
    elif average == 'binary':
        axis = None
    else:
        raise ValueError(
            "Invalid value for average. Must be 'micro', 'macro', 'samples', 'weighted', or 'binary'."
        )

    scores = TP.sum(axis) / (TP + 0.5 * (FP + FN)).sum(axis)

    if average == 'weighted':
        score = np.average(scores, weights=weights)
    else:
        score = np.nanmean(scores)

    return score


def modified_f1_score(
    Y_test,
    Y_pred,
    delta=0.2,
    average='samples',
    weights=None,
):
    assert Y_test.shape == Y_pred.shape

    # Calculate error
    E = np.minimum(abs(Y_pred - Y_test) / (Y_test + 1e-10), 1)

    ATP = (Y_pred > 0) & (Y_test > 0) & (E <= delta)
    ITP = (Y_pred > 0) & (Y_test > 0) & (E > delta)
    FP = (Y_pred > 0) & (Y_test == 0)
    FN = (Y_pred <= 0) & (Y_test > 0)

    if average == 'micro':
        axis = None
    elif average == 'macro':
        axis = 0
    elif average == 'samples':
        axis = 1
    elif average == 'weighted':
        axis = 0
        assert weights is not None
        assert len(weights) == Y_test.shape[1]
    elif average == 'binary':
        axis = None
    else:
        raise ValueError(
            "Invalid value for average. Must be 'micro', 'macro', 'samples', 'weighted', or 'binary'."
        )

    scores = ATP.sum(axis) / (ATP + ITP + 0.5 * (FP + FN)).sum(axis)

    if average == 'weighted':
        score = np.average(scores, weights=weights)
    else:
        score = np.nanmean(scores)

    return score


def jaccard_score(Y_test, Y_pred, average='samples', weights=None):
    assert Y_test.shape == Y_pred.shape

    # Calculate error
    TP = (Y_pred > 0) & (Y_test > 0)
    FP = (Y_pred > 0) & (Y_test == 0)
    FN = (Y_pred <= 0) & (Y_test > 0)

    if average == 'micro':
        axis = None
    elif average == 'macro':
        axis = 0
    elif average == 'samples':
        axis = 1
    elif average == 'weighted':
        axis = 0
        assert weights is not None
        assert len(weights) == Y_test.shape[1]
    elif average == 'binary':
        axis = None
    else:
        raise ValueError(
            "Invalid value for average. Must be 'micro', 'macro', 'samples', 'weighted', or 'binary'."
        )

    scores = TP.sum(axis) / (TP + FP + FN).sum(axis)

    if average == 'weighted':
        score = np.average(scores, weights=weights)
    else:
        score = np.nanmean(scores)

    return score


def modified_jaccard_score(
    Y_test,
    Y_pred,
    delta=20,
    average='samples',
    weights=None,
    rel=False,
):
    assert Y_test.shape == Y_pred.shape

    # Calculate error
    if rel:
        E = np.minimum(abs(Y_pred - Y_test) / (Y_test + 1e-10), 1)
    else:
        E = abs(Y_pred - Y_test)

    ATP = (Y_pred > 0) & (Y_test > 0) & (E <= delta)
    ITP = (Y_pred > 0) & (Y_test > 0) & (E > delta)
    FP = (Y_pred > 0) & (Y_test == 0)
    FN = (Y_pred <= 0) & (Y_test > 0)

    if average == 'micro':
        axis = None
    elif average == 'macro':
        axis = 0
    elif average == 'samples':
        axis = 1
    elif average == 'weighted':
        axis = 0
        assert weights is not None
        assert len(weights) == Y_test.shape[1]
    elif average == 'binary':
        axis = None
    else:
        raise ValueError(
            "Invalid value for average. Must be 'micro', 'macro', 'samples', 'weighted', or 'binary'."
        )

    scores = ATP.sum(axis) / (ATP + ITP + FP + FN).sum(axis)

    if average == 'weighted':
        score = np.average(scores, weights=weights)
    else:
        score = np.nanmean(scores)

    return score
