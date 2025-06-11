import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

def evaluate(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_true, y_pred)
    return acc, loss, cm

