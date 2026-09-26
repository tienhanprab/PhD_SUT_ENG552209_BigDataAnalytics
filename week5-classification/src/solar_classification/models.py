"""Deterministic categorical OneR and pure-rule PRISM.

PRISM uses positive coverage to break precision ties, then input feature order
and sorted category order. Impure terminal candidates are rejected; their
positives are deferred to the default, while ALL negatives remain available.
This explicit noise policy prevents infinite loops on contradictory categories.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd


def majority(y):
    values, counts = np.unique(y, return_counts=True)
    return int(values[np.argmax(counts)])


def validate(X, y):
    if len(X) == 0 or len(X) != len(y) or X.isna().any().any():
        raise ValueError("Require nonempty, aligned, complete categorical data")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Expected binary labels 0/1")


@dataclass
class Rule:
    conditions: tuple
    label: int
    support: int
    correct: int

    def mask(self, X):
        mask = np.ones(len(X), dtype=bool)
        for feature, value in self.conditions:
            mask &= X[feature].to_numpy() == value
        return mask

    def text(self):
        antecedent = " AND ".join(f"{f} = {v}" for f, v in self.conditions) or "TRUE"
        return f"IF {antecedent} THEN {'Fault' if self.label else 'Normal'}"


class OneR:
    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        validate(X, y)
        self.default_ = majority(y)
        self.candidates_ = []
        best_error = len(y) + 1
        for feature in X.columns:
            rules, error = [], 0
            for value in sorted(X[feature].unique()):
                subset = y[X[feature].to_numpy() == value]
                label = majority(subset)
                correct = int((subset == label).sum())
                rules.append(Rule(((feature, value),), label, len(subset), correct))
                error += len(subset) - correct
            self.candidates_.append(dict(feature=feature, errors=error, error_rate=error / len(y)))
            if error < best_error:
                best_error, self.feature_, self.rules_ = error, feature, rules
        return self

    def predict(self, X):
        pred = np.full(len(X), self.default_, dtype=int)
        for rule in self.rules_:
            pred[rule.mask(X)] = rule.label
        return pred


class Prism:
    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        validate(X, y)
        self.default_ = majority(y)
        self.rules_, self.rejected_ = [], []
        arrays = {f: X[f].to_numpy() for f in X.columns}
        for label in sorted(np.unique(y)):
            remaining = np.ones(len(y), dtype=bool)
            while np.any(remaining & (y == label)):
                covered, conditions = remaining.copy(), []
                available = list(X.columns)
                while np.any(covered & (y != label)) and available:
                    best = None
                    for feature in available:
                        for value in sorted(np.unique(arrays[feature][covered])):
                            mask = covered & (arrays[feature] == value)
                            total = int(mask.sum())
                            positive = int((mask & (y == label)).sum())
                            if not positive:
                                continue
                            score = (Fraction(positive, total), positive)
                            if best is None or score > best[0]:
                                best = (score, feature, value, mask)
                    if best is None:
                        break
                    _, feature, value, covered = best
                    conditions.append((feature, value))
                    available.remove(feature)
                positive_mask = covered & (y == label)
                if not positive_mask.any():
                    raise RuntimeError("PRISM failed to make progress")
                if np.all(y[covered] == label):
                    # Full original training support, rather than residual support.
                    rule = Rule(tuple(conditions), int(label), 0, 0)
                    full = rule.mask(X)
                    rule.support = int(full.sum())
                    rule.correct = int((y[full] == label).sum())
                    assert rule.support == rule.correct
                    self.rules_.append(rule)
                else:
                    self.rejected_.append(dict(label=int(label), positives=int(positive_mask.sum()),
                                              conditions=len(conditions)))
                remaining[positive_mask] = False
        return self

    def predict_details(self, X):
        votes = np.zeros((len(X), 2), dtype=bool)
        for rule in self.rules_:
            votes[rule.mask(X), rule.label] = True
        matched = votes.sum(axis=1)
        pred = np.full(len(X), self.default_, dtype=int)
        single = matched == 1
        pred[single] = votes[single].argmax(axis=1)
        return pred, matched == 0, matched == 2

    def predict(self, X):
        return self.predict_details(X)[0]


class TertileDiscretizer:
    """Training-only quantile cut points; right-closed intervals, infinite tails.

    Equal quantiles collapse to fewer bins (2: Low/High; 1: Medium).
    'Medium' is a relative rank, never an engineering normal-voltage band.
    """
    def fit(self, X):
        self.medians_ = X.median()
        if self.medians_.isna().any():
            raise ValueError("A feature has no observed training values")
        filled = X.fillna(self.medians_)
        self.cuts_ = {}
        for feature in X:
            values = filled[feature]
            cuts = np.unique(values.quantile([1 / 3, 2 / 3]).to_numpy())
            self.cuts_[feature] = cuts[(cuts > values.min()) & (cuts < values.max())]
        return self

    def transform(self, X):
        result = {}
        for feature, cuts in self.cuts_.items():
            labels = {0: ["Medium"], 1: ["Low", "High"], 2: ["Low", "Medium", "High"]}[len(cuts)]
            values = X[feature].fillna(self.medians_[feature]).to_numpy()
            result[feature] = np.asarray(labels)[np.searchsorted(cuts, values, side="left")]
        return pd.DataFrame(result, index=X.index)
