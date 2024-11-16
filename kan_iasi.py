import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization,LSTM,Attention, Conv2D, MaxPooling2D, SimpleRNN,TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, cheby1
import pywt
import sklearn		
from loadData import loadData  # type: ignore
from processData import processData  	 # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint	
from tensorflow.keras.optimizers import Adam	
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
from kan import KAN
import torch
import os



# antrenare
data_directory = 'train_data'
load_Data = loadData(data_directory)
dataStore, labels = load_Data.loadData_armthreeClasses()




fs = 512 
window_length = 2
overlap = 1
process_Data = processData(dataStore,labels,fs,overlap,window_length)
X,Y = process_Data.extractArmFeatures()


X = np.array(X,dtype=np.float32)
Y = np.array(Y)


shape = X.shape
data_reshaped = X.reshape(-1, shape[-1])
scaler = sklearn.preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data_reshaped)
X = data_scaled.reshape(shape)
X_reshaped = X.reshape(X.shape[0], -1)
X_train = X
y_train = Y



# validare 
data_directory = 'valid_data'
load_Data = loadData(data_directory)
dataStore, labels = load_Data.loadData_armthreeClasses()



fs = 512 
window_length = 2
overlap = 1
process_Data = processData(dataStore,labels,fs,overlap,window_length)
X,Y = process_Data.extractArmFeatures()
X = np.array(X,dtype=np.float32)
Y = np.array(Y)
shape = X.shape
data_reshaped = X.reshape(-1, shape[-1])
scaler = sklearn.preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data_reshaped)
X = data_scaled.reshape(shape)
X_reshaped = X.reshape(X.shape[0], -1)
X_valid = X
y_valid = Y




print(X_train.shape)
print(y_train.shape)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_valid.reshape(X_valid.shape[0], -1)


dataset = {
    'train_input': torch.tensor(X_train_flattened, dtype=torch.float32),
    'train_label': torch.tensor(y_train, dtype=torch.long),
    'test_input': torch.tensor(X_test_flattened, dtype=torch.float32),
    'test_label': torch.tensor(y_valid, dtype=torch.long)
}


# Crearea modelului KAN
ckpt_path = os.path.join('.','model')
model = KAN(width=[40, 10, 10, 3], grid=5, k=3, seed=0, ckpt_path=ckpt_path)

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# Funcțiile pentru calcularea acurateții
def train_acc():
    accuracy = torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())
    train_acc_history.append(accuracy.item())
    return accuracy

def test_acc():
    accuracy = torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
    test_acc_history.append(accuracy.item())
    return accuracy

# Antrenarea modelului
result = model.fit(
    dataset, 
    opt="Adam", 
    steps=300, 
    lamb=0.0001, 
    lamb_entropy=0.1,
    metrics=(train_acc, test_acc), 
    loss_fn=torch.nn.CrossEntropyLoss()
)

# Salvează istoricul `loss` (train și test) din `result`
train_loss_history = result['train_loss']
test_loss_history = result['test_loss']

# Plotarea graficelor de `loss` și `acuratețe`
plt.figure(figsize=(12, 5))

# Plotarea pierderii (loss)
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Steps')
plt.legend()

# Plotarea acurateții
plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Steps')
plt.legend()

plt.tight_layout()
plt.show()

mode = "manual" # "manual"
if mode == "manual":
    # manual mode
    model.fix_symbolic(1,0,0,'exp')
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)


train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# Funcțiile pentru calcularea acurateții
def train_acc():
    accuracy = torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())
    train_acc_history.append(accuracy.item())
    return accuracy

def test_acc():
    accuracy = torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
    test_acc_history.append(accuracy.item())
    return accuracy

# Antrenarea modelului
result = model.fit(
    dataset, 
    opt="Adam", 
    steps=100, 
    metrics=(train_acc, test_acc), 
    loss_fn=torch.nn.CrossEntropyLoss()
)

# Salvează istoricul `loss` (train și test) din `result`
train_loss_history = result['train_loss']
test_loss_history = result['test_loss']

# Plotarea graficelor de `loss` și `acuratețe`
plt.figure(figsize=(12, 5))

# Plotarea pierderii (loss)
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Steps')
plt.legend()

# Plotarea acurateții
plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Steps')
plt.legend()

plt.tight_layout()
plt.show()