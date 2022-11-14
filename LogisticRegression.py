import numpy as np


def generate_batches(X, y, batch_size):
    assert len(X) == len(y), "len X!=len y"
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    n_samples = X.shape[0]
    perm = np.random.permutation(len(X))
    for start in range(0, n_samples, batch_size):
        if(n_samples - start) >= batch_size:
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            yield X[batch_idx], y[batch_idx]


def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1./(1+np.exp(-h))


class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=100, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            self.w = np.random.randn(k + 1)
        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        losses = []
        for epoch in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                predictions = self._predict_proba_internal(X_batch)
                loss = self._loss(y_batch, predictions)
                assert (np.array(loss).shape == tuple()), "Loss must be scalar"
                losses.append(loss)
                grad = self.get_grad(X_batch, y_batch, predictions)
                self.w -= lr * grad
        return losses

    def get_grad(self, X_batch, y_batch, predictions, lr=1):
        grad_basic = lr * (np.transpose(X_batch) @ (predictions - y_batch))
        assert grad_basic.shape == (X_batch.shape[1],), 'Gradients mast be column of k_features+1 elements'
        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def _loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))