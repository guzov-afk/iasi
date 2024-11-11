import numpy as np
import utils_iasi as utils




class processData():
    def __init__(self, dataStore, labels, frequencySampling, overlapping, window_length):
        self.dataStore = dataStore
        self.labels = labels
        self.frequencySampling = frequencySampling
        self.overlapping = overlapping
        self.window_length = window_length

    def extractArmFeatures(self):
        X, Y = [], []
        N = int(self.frequencySampling * self.window_length)

        # Parcurge fiecare set de date și etichetă
        for count, d in enumerate(self.dataStore):
            var = self.labels[count]
            start = 0

            # Parcurge ferestrele de date
            while start + N <= len(d[0]):
                features_channel = []

                for channel_data in d:
                    window = channel_data[start:start + N]
                    
                    # Calcularea caracteristicilor pentru fereastra curentă
                    rms = utils.RMS(window)
                    wl = utils.WL(window)
                    zcr = utils.ZCR(window)
                    mav = utils.MAV(window)
                    
                    
                    # Adăugarea caracteristicilor în listă
                    features_channel.extend([rms, wl, zcr, mav])

                # Adăugarea caracteristicilor și etichetei în setul de date final
                X.append(features_channel)
                Y.append(var)

                # Actualizează poziția de start pentru fereastra următoare
                start += int(N * (1 - self.overlapping))

        return np.array(X), np.array(Y)
