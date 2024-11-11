import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from loadDataIasi import loadData
from processDataIasi import processData
from kan import KAN

class DataProcessor:
    def __init__(self, data_directory, fs, window_length, overlap):
        self.data_directory = data_directory
        self.fs = fs
        self.window_length = window_length
        self.overlap = overlap
    
    def load_and_process_data(self):
        load_data = loadData(self.data_directory)
        dataStore, labels = load_data.loadData_armthreeClasses()
        process_data = processData(dataStore, labels, self.fs, self.overlap, self.window_length)
        X, Y = process_data.extractArmFeatures()
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y)
        
        # Standardize the data
        shape = X.shape
        data_reshaped = X.reshape(-1, shape[-1])
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)
        X = data_scaled.reshape(shape)
        
        return X, Y

class KANModel:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model = KAN(width=[40, 10, 10, 3], grid=5, k=3, seed=0, ckpt_path=ckpt_path)
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def prepare_data(self, X_train, y_train, X_test, y_test):
        self.dataset = {
            'train_input': torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32),
            'train_label': torch.tensor(y_train, dtype=torch.long),
            'test_input': torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32),
            'test_label': torch.tensor(y_test, dtype=torch.long)
        }
    
    def calculate_train_accuracy(self):
        accuracy = torch.mean((torch.argmax(self.model(self.dataset['train_input']), dim=1) == self.dataset['train_label']).float())
        self.train_acc_history.append(accuracy.item())
        return accuracy

    def calculate_test_accuracy(self):
        accuracy = torch.mean((torch.argmax(self.model(self.dataset['test_input']), dim=1) == self.dataset['test_label']).float())
        self.test_acc_history.append(accuracy.item())
        return accuracy

    def train(self, steps=300, learning_rate=0.0001, entropy_reg=0.1):
        result = self.model.fit(
            self.dataset, 
            opt="Adam", 
            steps=steps, 
            lamb=learning_rate, 
            lamb_entropy=entropy_reg,
            metrics=(self.calculate_train_accuracy, self.calculate_test_accuracy), 
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        
        # Save loss history from result
        self.train_loss_history = result['train_loss']
        self.test_loss_history = result['test_loss']
    
    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        
        # Plotting loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.test_loss_history, label='Test Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss over Steps')
        plt.legend()
        
        # Plotting accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='Train Accuracy')
        plt.plot(self.test_acc_history, label='Test Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy over Steps')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def set_symbolic_mode(self, mode="manual"):
        if mode == "manual":
            self.model.fix_symbolic(1, 0, 0, 'exp')
        elif mode == "auto":
            lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
            self.model.auto_symbolic(lib=lib)


def main():
    # Set parameters
    fs = 512
    window_length = 2
    overlap = 1
    ckpt_path = os.path.join('.', 'model')
    
    # Load and process training data
    train_data_processor = DataProcessor('train_data', fs, window_length, overlap)
    X_train, y_train = train_data_processor.load_and_process_data()
    
    # Load and process validation data
    valid_data_processor = DataProcessor('valid_data', fs, window_length, overlap)
    X_valid, y_valid = valid_data_processor.load_and_process_data()
    
    # Initialize and prepare KAN model
    kan_model = KANModel(ckpt_path)
    kan_model.prepare_data(X_train, y_train, X_valid, y_valid)
    
    # Train the model and plot results
    kan_model.train(steps=300, learning_rate=0.0001, entropy_reg=0.1)
    kan_model.plot_metrics()
    
    # Set symbolic mode
    kan_model.set_symbolic_mode("manual")

if __name__ == "__main__":
    main()
