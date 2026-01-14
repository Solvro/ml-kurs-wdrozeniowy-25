import pandas as pd
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class TrainingClass:

  def __init__(self, model, device, loss_fn, optimizer, early_stopping=False, scheduler=None, patience=10):
    super().__init__()
    self.model = model.to(device)
    self.patience = patience
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.device = device
    self.early_stopping = early_stopping
    self.scheduler = scheduler

  def training_loop(self, epochs, model_name, trainloader, validationloader, testloader):
    # inicjalizacja list zwracanych przez funkcję
    training_losses = []
    validation_losses = []

    train_metrics = []
    validation_metrics = []

    training_accuracies = []
    validation_accuracies = []

    best_loss = float('inf')
    wait = 0
    for epoch in tqdm(range(epochs)):
      # trening
      self.train(trainloader)
      # ewaluacja zbioru treningowego
      train_loss, metrics = self.evaluate(trainloader)
      training_losses.append(train_loss)
      train_df, train_acc = self.create_metrics_dataframe(metrics)
      training_accuracies.append(train_acc)
      train_metrics.append(train_df)
      # walidacja zbiorem walidacyjnym
      loss, metrics = self.evaluate(validationloader)
      validation_losses.append(loss)
      val_df, val_acc = self.create_metrics_dataframe(metrics)
      validation_accuracies.append(val_acc)
      validation_metrics.append(val_df)

      # scheduler
      if self.scheduler is not None:
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.4f}")

      # zapamiętanie najlepszego modelu
      if loss < best_loss:
        best_loss = loss
        wait = 0
        torch.save(self.model.state_dict(), f'/content/drive/MyDrive/mlp-models/{model_name}_weights.pth')
      else:
        wait += 1
        # early stopping
        if self.early_stopping and wait >= self.patience:
          print(f'Early stopping at epoch {epoch+1}')
          break
      # wyświetlanie statusu
      if epoch % 2 == 0: # co druga epoka aby nie zaśmieciać notatnika output'em xd
        print(f'epoch: {epoch+1}, train loss: {train_loss:.3f}, validation loss: {loss:.3f}')

    # końcowa ewaluacja zbiorem testowym
    test_loss, _ = self.evaluate(testloader)
    print(f'test loss: {test_loss:.3f}')

    return training_losses, validation_losses, train_metrics, validation_metrics, training_accuracies, validation_accuracies

  # funkcja do treningu modelu
  def train(self, dataloader):
    self.model.train()
    for inputs, labels in dataloader:
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      self.optimizer.zero_grad()
      predictions = self.model(inputs)
      loss = self.loss_fn(predictions, labels)

      loss.backward()
      self.optimizer.step()

  # funkcja do oceny modelu na wybranym dataloaderze i pozyskania metryk
  def evaluate(self, dataloader):
    self.model.eval()
    all_labels = []       # lista wartości rzeczywistych
    all_predictions = []  # lista wartości przewidzianych przez model
    with torch.no_grad():
      total_loss = 0.0
      for inputs, labels in dataloader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        # sumowanie straty z każdego batch'a dataloader'a
        total_loss += self.loss_fn(outputs, labels).item()

        _, predicted = torch.max(outputs.data, 1) # wydobycie klasy o największej wartości (sama wartość jest pomijana ("_"))
        all_labels.append(labels)
        all_predictions.append(predicted)

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_predictions = torch.cat(all_predictions).cpu().numpy()

    # wydobycie średniej
    return total_loss / len(dataloader), classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0, output_dict=True)

  # funkcja do przetworzenia classification_report z funkcji get_metrics do postaci DataFrame
  def create_metrics_dataframe(self, metrics):
    data = {}
    acc = metrics.pop('accuracy') # pobranie accuracy
    for key, value in metrics.items():
      # pobranie informacji o precision, recall, f1 per klasa
      if key in target_names:
        for k, v in value.items():
          if k != 'support':
            if k not in data:
              data[k] = {}
            data[k][key] = v
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'metric'
    return df, acc

# kod dla pojedynczego subplot
def plot_loss_and_accuracy(tr_list, val_list, function_name, plot_num=1):
  plt.subplot(1, 2, plot_num)
  plt.title(f'Wykres {function_name}')
  plt.plot(tr_list, label='trening')
  plt.plot(val_list, label='walidacja')
  plt.xlabel('Epoka')
  plt.legend()
  plt.grid(True)

# utworzenie wizualizacji dla funkcji straty i accuracy
def create_loss_and_accuracy_plot(tr_losses, val_losses, tr_accuracies, val_accuracies):
  plt.figure(figsize=(15, 5))
  plot_loss_and_accuracy(tr_losses, val_losses, 'funkcji straty', 1)
  plot_loss_and_accuracy(tr_accuracies, val_accuracies, 'dokładności', 2)

train_colors = ['tomato', 'limegreen', 'cornflowerblue']
validation_colors = ['darkred', 'darkgreen', 'darkblue']

def plot_metrics(dataframes, dataframe_type, colors):
  for idx, col in enumerate(dataframes[0].columns): # pętla do utworzenia dla każdej klasy osobnego subplotu
    plt.subplot(5, 2, idx+1)
    # zebranie metryk z listy DataFrame do odpowiednich list
    precision = []
    recall = []
    f1 = []
    for metrics in dataframes:
      precision.append(metrics.loc['precision', col])
      recall.append(metrics.loc['recall', col])
      f1.append(metrics.loc['f1-score', col])
    # wykres
    plt.title(f'Metryki dla klasy {col}')
    plt.xlabel('Epoka')
    plt.ylim(-0.05, 1.05)
    plt.plot(precision, label=f'precision - {dataframe_type}', color=colors[0]) # czerwony dla precision
    plt.plot(recall, label=f'recall - {dataframe_type}', color=colors[1]) # zielony dla recall
    plt.plot(f1, label=f'f1-score - {dataframe_type}', color=colors[2]) # niebieski dla f1
    plt.legend()
    plt.grid(True)

# utworzenie wizualizacji dla pozostałych metryk (precision, recall, f1)
def create_metrics_plot(tr_metrics, val_metrics):
  plt.figure(figsize=(15, 20))
  plot_metrics(tr_metrics, 'trening', train_colors)
  plot_metrics(val_metrics, 'walidacja', validation_colors)
  plt.tight_layout()
  plt.show()