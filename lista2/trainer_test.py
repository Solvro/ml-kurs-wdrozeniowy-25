from trainer import Trainer
import numpy as np

if __name__ == "__main__":
  from perceptron import Perceptron
  
  # Initialize perceptron for tests
  perceptron = Perceptron(wSize=2)
  trainer = Trainer(perceptron=perceptron, n_epochs=1, learning_rate=0.01)

  print("=" * 50)
  print("TESTING LOSS FUNCTION")
  print("=" * 50)
  
  # Test 1: basic loss correctness
  A = np.array([0.9, 0.8, 0.1])
  y = np.array([1, 1, 0])
  loss_val = trainer.loss(A, y)
  expected = np.mean(-(y * np.log(A) + (1 - y) * np.log(1 - A)))
  print(f"Loss: {loss_val:.8f}, expected: {expected:.8f}")
  assert np.isclose(loss_val, expected), "Loss value differs from expected"
  print("✓ Loss function numeric test passed\n")

  # Test 2: loss non-negativity and finiteness
  A2 = np.array([0.5, 0.5])
  y2 = np.array([0, 1])
  loss_val2 = trainer.loss(A2, y2)
  assert np.isfinite(loss_val2) and loss_val2 >= 0, "Loss should be finite and non-negative"
  print(f"Loss with A=0.5: {loss_val2:.8f}")
  print("✓ Loss non-negativity test passed\n")

  # Test 3: perfect predictions should have near-zero loss
  A3 = np.array([0.9999, 0.9999, 0.0001, 0.0001])
  y3 = np.array([1, 1, 0, 0])
  loss_val3 = trainer.loss(A3, y3)
  print(f"Loss with near-perfect predictions: {loss_val3:.8f}")
  assert loss_val3 < 0.01, "Loss should be very small for near-perfect predictions"
  print("✓ Perfect prediction loss test passed\n")

  print("=" * 50)
  print("TESTING BACKWARD PROPAGATION")
  print("=" * 50)

  # Test 4: backward single-sample case
  X_single = np.array([[1.0], [2.0]])  # shape (features=2, samples=1)
  A_single = np.array([0.7])
  y_single = np.array([1.0])
  dw_single, db_single = trainer.backward(X=X_single, A=A_single, y=y_single)
  expected_dw_single = X_single @ (A_single - y_single).T
  if expected_dw_single.ndim == 1:
    expected_dw_single = expected_dw_single.reshape(-1, 1)
  expected_dw_single = expected_dw_single / X_single.shape[1]
  expected_db_single = np.mean((A_single - y_single) * 1)
  print(f"Backward single-sample dw shape: {dw_single.shape}")
  print(f"Backward single-sample dw: {dw_single.flatten()}")
  print(f"Expected dw: {expected_dw_single.flatten()}")
  print(f"Backward single-sample db: {db_single:.8f}")
  print(f"Expected db: {expected_db_single:.8f}")
  assert dw_single.shape == (2, 1), f"Backward gradient shape mismatch for single sample: got {dw_single.shape}"
  assert np.allclose(dw_single, expected_dw_single), "Backward gradient value mismatch for single sample"
  assert np.isclose(db_single, expected_db_single), "Backward bias gradient mismatch"
  print("✓ Backward single-sample test passed\n")

  # Test 5: backward multi-sample case
  X_multi = np.array([[1.0, 1.0, 0.0],
                      [0.0, 1.0, 1.0]])  # shape (features=2, samples=3)
  A_multi = np.array([0.9, 0.2, 0.1])
  y_multi = np.array([1, 0, 0])
  dw_multi, db_multi = trainer.backward(X=X_multi, A=A_multi, y=y_multi)
  expected_dw_multi = X_multi @ (A_multi - y_multi).T
  if expected_dw_multi.ndim == 1:
    expected_dw_multi = expected_dw_multi.reshape(-1, 1)
  expected_dw_multi = expected_dw_multi / X_multi.shape[1]
  expected_db_multi = np.mean((A_multi - y_multi) * 1)
  print(f"Backward multi-sample dw shape: {dw_multi.shape}")
  print(f"Backward multi-sample dw: {dw_multi.flatten()}")
  print(f"Expected dw: {expected_dw_multi.flatten()}")
  print(f"Backward multi-sample db: {db_multi:.8f}")
  print(f"Expected db: {expected_db_multi:.8f}")
  assert dw_multi.shape == (2, 1), f"Backward gradient shape mismatch for multi-sample case: got {dw_multi.shape}"
  assert np.allclose(dw_multi, expected_dw_multi), "Backward gradient value mismatch for multi-sample case"
  assert np.isclose(db_multi, expected_db_multi), "Backward bias gradient mismatch for multi-sample"
  print("✓ Backward multi-sample test passed\n")

  print("=" * 50)
  print("TESTING WEIGHT UPDATES")
  print("=" * 50)

  # Test 6: weight update correctness
  perceptron_test = Perceptron(wSize=2)
  perceptron_test.w = np.array([[1.0], [2.0]])
  perceptron_test.b = 0.5
  trainer_test = Trainer(perceptron=perceptron_test, n_epochs=1, learning_rate=0.1)
  
  dw = np.array([[0.2], [0.3]])
  db = 0.1
  
  old_w = perceptron_test.w.copy()
  old_b = perceptron_test.b
  
  trainer_test.update_weights(dw, db)
  
  expected_w = old_w - 0.1 * dw
  expected_b = old_b - 0.1 * db
  
  print(f"Old weights: {old_w.flatten()}, New weights: {perceptron_test.w.flatten()}")
  print(f"Expected weights: {expected_w.flatten()}")
  print(f"Old bias: {old_b:.4f}, New bias: {perceptron_test.b:.4f}")
  print(f"Expected bias: {expected_b:.4f}")
  
  assert np.allclose(perceptron_test.w, expected_w), "Weight update incorrect"
  assert np.isclose(perceptron_test.b, expected_b), "Bias update incorrect"
  print("✓ Weight update test passed\n")

  print("=" * 50)
  print("TESTING PREDICT FUNCTION")
  print("=" * 50)

  # Test 7: predict function
  X_pred = np.array([[0.5, -0.5, 1.0],
                     [0.5, 0.5, -1.0]])
  perceptron_test.w = np.array([[1.0], [1.0]])
  perceptron_test.b = 0.0
  
  predictions = trainer_test.predict(X_pred)
  print(f"Predictions shape: {predictions.shape}")
  print(f"Predictions: {predictions.flatten()}")
  assert predictions.shape[1] == 3, "Prediction shape mismatch"
  assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"
  print("✓ Predict function test passed\n")

  print("=" * 50)
  print("TESTING ACCURACY FUNCTION")
  print("=" * 50)

  # Test 8: accuracy function
  y_pred = np.array([[1, 0, 1, 0]])
  y_true = np.array([[1, 0, 0, 0]])
  acc = trainer_test.accuracy(y_pred, y_true)
  expected_acc = 0.75
  print(f"Accuracy: {acc:.2f}, Expected: {expected_acc:.2f}")
  assert np.isclose(acc, expected_acc), "Accuracy calculation incorrect"
  print("✓ Accuracy function test passed\n")

  # Test 9: perfect accuracy
  y_pred_perfect = np.array([[1, 1, 0, 0]])
  y_true_perfect = np.array([[1, 1, 0, 0]])
  acc_perfect = trainer_test.accuracy(y_pred_perfect, y_true_perfect)
  print(f"Perfect accuracy: {acc_perfect:.2f}")
  assert acc_perfect == 1.0, "Perfect accuracy should be 1.0"
  print("✓ Perfect accuracy test passed\n")

  print("=" * 50)
  print("TESTING TRAINING LOOP")
  print("=" * 50)

  # Test 10: training decreases loss
  np.random.seed(42)
  perceptron_train = Perceptron(wSize=2)
  trainer_train = Trainer(perceptron=perceptron_train, n_epochs=100, learning_rate=0.1)
  
  # Simple linearly separable dataset
  X_train = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])
  y_train = np.array([[0, 0, 0, 1]])  # AND gate
  
  X_test = X_train.copy()
  y_test = y_train.copy()
  
  initial_loss = trainer_train.loss(perceptron_train.forward(X_train), y_train)
  print(f"Initial loss: {initial_loss:.4f}")
  
  trainer_train.train(X_train, y_train, X_test, y_test)
  
  final_loss = trainer_train.train_loss_track[-1]
  print(f"Final loss: {final_loss:.4f}")
  
  assert final_loss < initial_loss, "Training should decrease loss"
  assert len(trainer_train.train_loss_track) == 100, "Loss track length should equal n_epochs"
  print("✓ Training loop test passed\n")

  print("=" * 50)
  print("ALL TESTS PASSED! ✓")
  print("=" * 50)