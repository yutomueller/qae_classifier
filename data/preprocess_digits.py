import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_and_preprocess(classes=[0,2,4,6], train_size=75, test_size=50):
    digits = load_digits()
    idx = np.isin(digits.target, classes)
    X = digits.images[idx]
    y = digits.target[idx]
    # flatten and normalize
    X = X.reshape(-1, 8*8).astype(np.float32)
    X = X / X.max(axis=1, keepdims=True)
    # remap labels to 0,1,2,3
    class_map = {c: i for i, c in enumerate(classes)}
    y = np.vectorize(class_map.get)(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=42)
    np.savez('digits_preprocessed.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print('Saved to digits_preprocessed.npz')

if __name__ == '__main__':
    load_and_preprocess()

