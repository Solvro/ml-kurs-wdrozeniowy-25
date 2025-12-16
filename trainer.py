import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

class Trainer:

  def __init__(self, perceptron: Perceptron, n_epochs: int, learning_rate: float):
    self.perceptron = perceptron
    self.n_epochs = n_epochs
    self.learning_rate = learning_rate
    self.train_loss_track = []
    self.val_loss_track = []
    self.train_accuracy_track = []
    self.val_accuracy_track = []

  def loss(self, A: np.array, y: np.ndarray): 
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1-epsilon)

    sample_losses = -(y * np.log(A) + (1 - y) * np.log(1 - A))

    return np.mean(sample_losses)

  def backward(self, X: np.ndarray, A: np.array, y: np.ndarray):
    dw_sum = X @ (A-y).T
    
    # Ensure dw has shape (n_features, 1)
    if dw_sum.ndim == 1:
      dw_sum = dw_sum.reshape(-1, 1)

    db = np.mean((A - y) * 1)

    return (dw_sum / X.shape[1]), db

  def update_weights(self, dw, db):
    new_w = self.perceptron.w - self.learning_rate * dw
    new_b = self.perceptron.b - self.learning_rate * db

    self.perceptron.w = new_w
    self.perceptron.b = new_b

  def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    if y_train.ndim == 1:
      y_train = y_train.reshape(1, -1)
    
    for epoch in range (self.n_epochs):
        A_train = self.perceptron.forward(X=x_train)
        A_val = self.perceptron.forward(X=x_val)

        train_loss = self.loss(A=A_train, y=y_train)
        val_loss = self.loss(A=A_val, y=y_val)

        self.train_loss_track.append(train_loss)
        self.val_loss_track.append(val_loss)

        dw, db = self.backward(X=x_train, A=A_train, y=y_train)

        self.update_weights(dw=dw, db=db)

        y_pred_train = self.predict(X=x_train)
        train_accuracy = self.accuracy(y_pred=y_pred_train, y_true=y_train)

        y_pred_val = self.predict(X=x_val)
        val_accuracy = self.accuracy(y_pred=y_pred_val, y_true=y_val)

        self.train_accuracy_track.append(train_accuracy)
        self.val_accuracy_track.append(val_accuracy)

        if epoch % 100 == 0:
          self.print_stats(epoch=epoch, L=train_loss, acc=train_accuracy)

  def predict(self, X: np.ndarray) -> np.ndarray:
    res = self.perceptron.forward(X=X)

    return np.round(res)

  def accuracy(self, y_pred, y_true):
    return np.mean(y_pred == y_true)
  
  def plot_metrics(self):
    epochs = range(1, len(self.train_loss_track) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Chart
    ax1.plot(epochs, self.train_loss_track, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, self.val_loss_track, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Chart
    ax2.plot(epochs, self.train_accuracy_track, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, self.val_accuracy_track, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
  
  def print_stats(self, epoch: int, L: float, acc: float):
    print(f"Epoch {epoch}: Loss {L:.4f}, Accuracy: {acc * 100:.1f}%")


