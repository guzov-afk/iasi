import numpy as np
import os
import random
import utils_iasi as utils

class loadData:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_data(self, filename):
        return np.load(filename)

    def process_channels(self, file_data):
        
        channels = [file_data[i].astype(int) - 128 for i in range(8)]
        return channels

    def loadData_armthreeClasses(self):
        data_store = []
        labels = []
        
        
        class_map = {"0": 0, "1": 1, "2": 2}

        for filename in os.listdir(self.data_directory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.data_directory, filename))
                parts = filename.split('_')
                cl = parts[2]
                
                
                if cl in class_map:
                    channels = self.process_channels(file_data)
                    data_store.append(channels)
                    labels.append(class_map[cl])

        return data_store, labels
