import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_true_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 30], [0, 30], color='red', linestyle='--')  # Plot y=x line for reference
    plt.xlabel('True Speed (m/s)')
    plt.ylabel('Predicted Speed (m/s)')
    plt.title('True vs. Predicted Speeds')
    plt.grid(True)
    plt.show()
