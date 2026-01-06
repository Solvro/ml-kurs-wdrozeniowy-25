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
    self.train_f1_track = []
    self.val_f1_track = []

  def loss(self, A: np.array, y: np.ndarray): 
    # prevent log(0) errors
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1-epsilon)

    # calculate loss
    sample_losses = -(y * np.log(A) + (1 - y) * np.log(1 - A)) # similar to an 'if' statement

    return np.mean(sample_losses)

  def backward(self, X: np.ndarray, A: np.array, y: np.ndarray):
    dw_sum = X @ (A-y).T # calculate gradient
    
    # Ensure dw has shape (n_features, 1)
    if dw_sum.ndim == 1:
      dw_sum = dw_sum.reshape(-1, 1)

    db = np.mean((A - y) * 1) # calculate gradient for bias

    return (dw_sum / X.shape[1]), db

  def update_weights(self, dw, db):
    new_w = self.perceptron.w - self.learning_rate * dw
    new_b = self.perceptron.b - self.learning_rate * db

    self.perceptron.w = new_w
    self.perceptron.b = new_b

  def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    # ensure correct dimensions
    if y_train.ndim == 1:
      y_train = y_train.reshape(1, -1)
    
    # begin the training
    for epoch in range (self.n_epochs):
        # forward pass
        A_train = self.perceptron.forward(X=x_train)
        A_val = self.perceptron.forward(X=x_val)

        # loss calculation
        train_loss = self.loss(A=A_train, y=y_train)
        val_loss = self.loss(A=A_val, y=y_val)

        # persist values posterior analysis
        self.train_loss_track.append(train_loss)
        self.val_loss_track.append(val_loss)

        # gradient calculation
        dw, db = self.backward(X=x_train, A=A_train, y=y_train)

        # weights update
        self.update_weights(dw=dw, db=db)

        # performance check
        y_pred_train = self.predict(X=x_train)
        train_f1 = self.f1_score(y_pred=y_pred_train, y_true=y_train)
        y_pred_val = self.predict(X=x_val)
        val_f1 = self.f1_score(y_pred=y_pred_val, y_true=y_val)

        # performance tracking
        self.train_f1_track.append(train_f1)
        self.val_f1_track.append(val_f1)

        if epoch % 100 == 0:
          self.print_stats(epoch=epoch, L=train_loss, f1=train_f1)

  def predict(self, X: np.ndarray) -> np.ndarray:
    res = self.perceptron.forward(X=X)

    return np.round(res)

  def f1_score(self, y_pred, y_true):
    # Flatten arrays if needed
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
  
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
    
    # F1 Score Chart
    ax2.plot(epochs, self.train_f1_track, 'b-', label='Training F1 Score', linewidth=2)
    ax2.plot(epochs, self.val_f1_track, 'r-', label='Validation F1 Score', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training vs Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
  
  def print_stats(self, epoch: int, L: float, f1: float):
    print(f"Epoch {epoch}: Loss {L:.4f}, F1 Score: {f1:.4f}")


