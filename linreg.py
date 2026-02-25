import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initialize the Linear Regression model.
        :param lr: Learning rate for gradient descent.
        :param n_iters: Number of iterations for gradient descent.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        :param X: Input features (numpy array).
        :param y: Target variable (numpy array).
        """
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.zeros((n_features, 1))

        # Reshape y to a column vector for matrix operations
        y = y.reshape(-1, 1)

        # Gradient Descent loop
        for _ in range(self.n_iters):
            # Calculate predictions (hypothesis: y = Xw + b)
            y_pred = np.dot(X, self.weights)

            # Calculate gradients
            # Gradient of loss w.r.t weights (dw)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))

            # Update weights
            self.weights -= self.lr * dw

    def predict(self, X):
        """
        Predict new values using the trained model.
        :param X: Input features (numpy array).
        :return: Predicted values.
        """
        y_pred = np.dot(X, self.weights)
        return y_pred
    
    def mse(self, X, y):
        y_hat = self.predict(X)
        y = y.reshape(-1,1)
        return np.mean((y_hat-y)**2)

# Example Usage:
if __name__ == '__main__':
    # Generate some synthetic data
    np.random.seed(42)
    n,p = (100,2)
    X = 2 * np.random.rand(n, p+1)
    X[:,0] = 1.0
    true_weights = np.array([[4.0], [3.0], [5.0]])
    y = np.dot(X, true_weights) + np.random.randn(100, 1)

    # Create and train the model
    regressor = LinearRegression(lr=0.01, n_iters=1000)
    regressor.fit(X, y)

    print(X.shape, y.shape)
    # Print the learned parameters
    print(f"Learned weights: {regressor.weights}")
    # Expected values are close to weights: [4.0,3.0,5.0]
    print(f"MSE: {regressor.mse(X,y):.3f}")
