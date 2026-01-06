import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

class Perceptron:

    def __init__(self, x_train, y_train, x_val, y_val):
        self.model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.x_train_tensor = torch.FloatTensor(x_train)
        self.y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        self.x_val_tensor = torch.FloatTensor(x_val)
        self.y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        self.loss_fn = nn.BCELoss()

        self.train_loss_track = []
        self.val_loss_track = []
        self.train_f1_track = []
        self.val_f1_track = []



    def train(self):
        for epoch in range(1000):
            # Forward pass
            y_pred_train = self.model(self.x_train_tensor)
            y_pred_val = self.model(self.x_val_tensor)
            
            # Calculate loss
            train_loss = self.loss_fn(y_pred_train, self.y_train_tensor)
            val_loss = self.loss_fn(y_pred_val, self.y_val_tensor)
            
            # Calculate F1 score
            train_pred_class = torch.round(y_pred_train)
            val_pred_class = torch.round(y_pred_val)
            train_f1 = self.f1_score(train_pred_class, self.y_train_tensor)
            val_f1 = self.f1_score(val_pred_class, self.y_val_tensor)
            
            # Track metrics
            self.train_loss_track.append(train_loss.item())
            self.val_loss_track.append(val_loss.item())
            self.train_f1_track.append(train_f1)
            self.val_f1_track.append(val_f1)
            
            # Backward pass
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Train F1 = {train_f1:.4f}")

        print(f"\nFinal Train F1 Score: {self.train_f1_track[-1]:.4f}")
        print(f"Final Val F1 Score: {self.val_f1_track[-1]:.4f}")



    def f1_score(self, y_pred, y_true):
        # Convert to numpy for calculation
        y_pred = y_pred.detach().cpu().numpy().flatten()
        y_true = y_true.detach().cpu().numpy().flatten()
        
        # Calculate True Positives, False Positives, False Negatives
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.train_loss_track) + 1)

        # Loss plot
        ax1.plot(epochs, self.train_loss_track, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_loss_track, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1 Score plot
        ax2.plot(epochs, self.train_f1_track, 'b-', label='Training F1 Score', linewidth=2)
        ax2.plot(epochs, self.val_f1_track, 'r-', label='Validation F1 Score', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Training vs Validation F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()



