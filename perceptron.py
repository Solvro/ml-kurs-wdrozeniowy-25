import numpy as np

class Perceptron:
  def __init__(self, wSize, b=None):
    rng = np.random.default_rng()
    self.w = rng.uniform(low=-0.01, high=0.01, size=(wSize, 1))
    self.b = b or rng.uniform(low=0.0, high=1.0)
    pass
  
  def Z(self, X: np.ndarray):
    return np.matrix_transpose(self.w) @ X + self.b
  
  def A(self, Z: int):
    return 1 / (1 + np.power(np.e, -Z))

  def forward(self, X: np.ndarray): 
    return self.A(self.Z(X=X))


if __name__ == '__main__':
    # Quick test
    p = Perceptron(wSize=3, b=0.0)
    X_dummy = np.random.rand(3, 5) # 3 features, 5 samples
    output = p.forward(X_dummy)

    print("Output shape:", output.shape)
    print("Output values:", output)