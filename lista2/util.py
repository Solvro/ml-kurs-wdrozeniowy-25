from trainer import Trainer

def test_trainers(trainers: list[Trainer], x_train,x_val, x_test, y_train, y_val, y_test):
    for trainer in trainers:
        print('=' * 50)
        print(f"Testing trainer with params \nlr: {trainer.learning_rate} \nepochs: {trainer.n_epochs} ")

        # Train the perceptron
        trainer.train(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)

        # Make predictions on test set
        y_pred = trainer.predict(X=x_test)
        test_f1 = trainer.f1_score(y_pred=y_pred, y_true=y_test)

        print(f"\nFinal Test F1 Score: {test_f1 * 100:.2f}%")
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred.flatten()[:100]}")

        trainer.plot_metrics()