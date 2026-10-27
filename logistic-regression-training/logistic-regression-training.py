import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))    

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    ## Initialize defaults of w and b
    m, n = X.shape # (m = Number of samples, n = number of features)
    print("X Shape: ", X.shape)

    ## 1. Initialize weights as zero
    w = np.zeros(n)
    print("W Shape: ", w)
    b = 0.0

    ## 2. Optimization loop
    for step in range(steps):
        ## 3. Use dot product for matrix multiplicatio 
        z = np.dot(X, w) + b
        predicted_proba = _sigmoid(z)
        
        ## 4. Compute gradients
        errors = predicted_proba-y
        dw = (1 / m) * np.dot(X.T, errors)
        db = (1 / m) * np.sum(errors)

        ## 5. Gradient descent step
        w-=dw*lr
        b-=db*lr

    return w, b
        
        
        