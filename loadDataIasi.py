import numpy as np
import os




class loadData():

    def __init__(self,dataDirectory):
        self.dataDirectory = dataDirectory

    def load_data(self,filename):
        data = np.load(filename)
        return data
    


    def loadData_armthreeClasses(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "0":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,30),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(0)
                elif cl == "1":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,30),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(1)
                elif cl == "2":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    #lis = [utils.emg_bandpass_filter(channel1,[20, 400],5000), utils.emg_bandpass_filter(channel2,[20, 400],5000), utils.emg_bandpass_filter(channel3,[20, 400],5000), utils.emg_bandpass_filter(channel4,[20, 400],5000), utils.emg_bandpass_filter(channel5,[20, 400],5000), utils.emg_bandpass_filter(channel6,[20, 400],5000), utils.emg_bandpass_filter(channel7,[20, 400],5000), utils.emg_bandpass_filter(channel8,[20, 400],5000)]
                    lis = [channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8]
                    #lis = [utils.ampl_filter(channel1,30),utils.ampl_filter(channel2,20),utils.ampl_filter(channel3,30),utils.ampl_filter(channel4,30),utils.ampl_filter(channel5,30),utils.ampl_filter(channel6,30),utils.ampl_filter(channel7,30),utils.ampl_filter(channel8,30)]

                    #random.shuffle(lis)
                    dataStore.append(lis)
                    labels.append(2)
        return dataStore,labels
    
    


    

