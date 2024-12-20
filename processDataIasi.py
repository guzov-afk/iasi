import numpy as np
import utils


class processData():
    def __init__(self,dataStore,labels,frequencySampling,overlapping,window_length):
        self.dataStore = dataStore
        self.labels = labels
        self.frequencySampling = frequencySampling
        self.overlapping = overlapping
        self.window_length = window_length
        



    def extractArmFeatures(self):
        rms_values_channel1 = []
        rms_values_channel2 = []
        rms_values_channel3 = []
        rms_values_channel4 = []
        rms_values_channel5 = []
        rms_values_channel6 = []
        rms_values_channel7 = []
        rms_values_channel8 = []

        wl_values_channel1 = []
        wl_values_channel2 = []
        wl_values_channel3 = []
        wl_values_channel4 = []
        wl_values_channel5 = []
        wl_values_channel6 = []
        wl_values_channel7 = []
        wl_values_channel8 = []

        ssc_values_channel1 = []
        ssc_values_channel2 = []
        ssc_values_channel3 = []
        ssc_values_channel4 = []
        ssc_values_channel5 = []
        ssc_values_channel6 = []
        ssc_values_channel7 = []
        ssc_values_channel8 = []

        zcr_values_channel1 = []
        zcr_values_channel2 = []
        zcr_values_channel3 = []
        zcr_values_channel4 = []
        zcr_values_channel5 = []
        zcr_values_channel6 = []
        zcr_values_channel7 = []
        zcr_values_channel8 = []

        mav_values_channel1 = []
        mav_values_channel2 = []
        mav_values_channel3 = []
        mav_values_channel4 = []
        mav_values_channel5 = []
        mav_values_channel6 = []
        mav_values_channel7 = []
        mav_values_channel8 = []

        e_values_channel1 = []
        e_values_channel2 = []
        e_values_channel3 = []
        e_values_channel4 = []
        e_values_channel5 = []
        e_values_channel6 = []
        e_values_channel7 = []
        e_values_channel8 = []

        hj_values_channel1 = []
        hj_values_channel2 = []
        hj_values_channel3 = []
        hj_values_channel4 = []
        hj_values_channel5 = []
        hj_values_channel6 = []
        hj_values_channel7 = []
        hj_values_channel8 = []

        skew_values_channel1 = []
        skew_values_channel2 = []
        skew_values_channel3 = []
        skew_values_channel4 = []
        skew_values_channel5 = []
        skew_values_channel6 = []
        skew_values_channel7 = []
        skew_values_channel8 = []

        iemg_values_channel1 = []
        iemg_values_channel2 = []
        iemg_values_channel3 = []
        iemg_values_channel4 = []
        iemg_values_channel5 = []
        iemg_values_channel6 = []
        iemg_values_channel7 = []
        iemg_values_channel8 = []


        

        X = []
        Y = []
        N = int(self.frequencySampling * self.window_length)
        start = 0

        for count,d in enumerate(self.dataStore):
            var = self.labels[count]
            
                
            start = 0
            while start + N <= len(d[0]):
                
                window1 = d[0][start:start + N]
                window2 = d[1][start:start + N]
                window3 = d[2][start:start + N]
                window4 = d[3][start:start + N]
                window5 = d[4][start:start + N]
                window6 = d[5][start:start + N]
                window7 = d[6][start:start + N]
                window8 = d[7][start:start + N]
                
                    

                #rms
                rms1 = utils.RMS(window1)
                rms_values_channel1.append(rms1)
                rms2 = utils.RMS(window2)
                rms_values_channel2.append(rms2)
                rms3 = utils.RMS(window3)
                rms_values_channel3.append(rms3)
                rms4 = utils.RMS(window4)
                rms_values_channel4.append(rms4)
                rms5 = utils.RMS(window5)
                rms_values_channel5.append(rms5)
                rms6 = utils.RMS(window6)
                rms_values_channel6.append(rms6)
                rms7 = utils.RMS(window7)
                rms_values_channel7.append(rms7)
                rms8 = utils.RMS(window8)
                rms_values_channel8.append(rms8)
                    

                #wl
                wl1 = utils.WL(window1)
                wl_values_channel1.append(wl1)
                wl2 = utils.WL(window2)
                wl_values_channel2.append(wl2)
                wl3 = utils.WL(window3)
                wl_values_channel3.append(wl3)
                wl4 = utils.WL(window4)
                wl_values_channel4.append(wl4)
                wl5 = utils.WL(window5)
                wl_values_channel5.append(wl5)
                wl6 = utils.WL(window6)
                wl_values_channel6.append(wl6)
                wl7 = utils.WL(window7)
                wl_values_channel7.append(wl7)
                wl8 = utils.WL(window8)
                wl_values_channel8.append(wl8)

                #SSC
                ssc1 = utils.SSC(window1,0)
                ssc_values_channel1.append(ssc1)
                ssc2 = utils.SSC(window2,0)
                ssc_values_channel2.append(ssc2)
                ssc3 = utils.SSC(window3,0)
                ssc_values_channel3.append(ssc3)
                ssc4 = utils.SSC(window4,0)
                ssc_values_channel4.append(ssc4)
                ssc5 = utils.SSC(window5,0)
                ssc_values_channel5.append(ssc5)
                ssc6 = utils.SSC(window6,0)
                ssc_values_channel6.append(ssc6)
                ssc7 = utils.SSC(window7,0)
                ssc_values_channel7.append(ssc7)
                ssc8 = utils.SSC(window8,0)
                ssc_values_channel8.append(ssc8)

                #ZCR
                zcr1 = utils.ZCR(window1)
                zcr_values_channel1.append(zcr1)
                zcr2 = utils.ZCR(window2)
                zcr_values_channel2.append(zcr2)
                zcr3 = utils.ZCR(window3)
                zcr_values_channel3.append(zcr3)
                zcr4 = utils.ZCR(window4)
                zcr_values_channel4.append(zcr4)
                zcr5 = utils.ZCR(window5)
                zcr_values_channel5.append(zcr5)
                zcr6 = utils.ZCR(window6)
                zcr_values_channel6.append(zcr6)
                zcr7 = utils.ZCR(window7)
                zcr_values_channel7.append(zcr7)
                zcr8 = utils.ZCR(window8)
                zcr_values_channel8.append(zcr8)
                    

                #MAV
                mav1 = utils.MAV(window1)
                mav_values_channel1.append(mav1)
                mav2 = utils.MAV(window2)
                mav_values_channel2.append(mav2)
                mav3 = utils.MAV(window3)
                mav_values_channel3.append(mav3)
                mav4 = utils.MAV(window4)
                mav_values_channel4.append(mav4)
                mav5 = utils.MAV(window5)
                mav_values_channel5.append(mav5)
                mav6 = utils.MAV(window6)
                mav_values_channel6.append(mav6)
                mav7 = utils.MAV(window7)
                mav_values_channel7.append(mav7)
                mav8 = utils.MAV(window8)
                mav_values_channel8.append(mav8)

                #energy
                e1 = utils.Energy(window1)
                e_values_channel1.append(e1)
                e2 = utils.Energy(window2)
                e_values_channel2.append(e2)
                e3 = utils.Energy(window3)
                e_values_channel3.append(e3)
                e4 = utils.Energy(window4)
                e_values_channel4.append(e4)
                e5 = utils.Energy(window5)
                e_values_channel5.append(e5)
                e6 = utils.Energy(window6)
                e_values_channel6.append(e6)
                e7 = utils.Energy(window7)
                e_values_channel7.append(e7)
                e8 = utils.Energy(window8)
                e_values_channel8.append(e8)


                hj1 = utils.HJ(window1)
                hj_values_channel1.append(hj1)
                hj2 = utils.HJ(window2)
                hj_values_channel2.append(hj2)
                hj3 = utils.HJ(window3)
                hj_values_channel3.append(hj3)
                hj4 = utils.HJ(window4)
                hj_values_channel4.append(hj4)
                hj5 = utils.HJ(window5)
                hj_values_channel5.append(hj5)
                hj6 = utils.HJ(window6)
                hj_values_channel6.append(hj6)
                hj7 = utils.HJ(window7)
                hj_values_channel7.append(hj7)
                hj8 = utils.HJ(window8)
                hj_values_channel8.append(hj8)

                skew1 = utils.Skewness(window1)
                skew_values_channel1.append(skew1)
                skew2 = utils.Skewness(window2)
                skew_values_channel2.append(skew2)
                skew3 = utils.Skewness(window3)
                skew_values_channel3.append(skew3)
                skew4 = utils.Skewness(window4)
                skew_values_channel4.append(skew4)
                skew5 = utils.Skewness(window5)
                skew_values_channel5.append(skew5)
                skew6 = utils.Skewness(window6)
                skew_values_channel6.append(skew6)
                skew7 = utils.Skewness(window7)
                skew_values_channel7.append(skew7)
                skew8 = utils.Skewness(window8)
                skew_values_channel8.append(skew8)

                iea1 = utils.iemg(window1)
                iemg_values_channel1.append(iea1)
                iea2 = utils.iemg(window2)
                iemg_values_channel2.append(iea2)
                iea3 = utils.iemg(window3)
                iemg_values_channel3.append(iea3)
                iea4 = utils.iemg(window4)
                iemg_values_channel4.append(iea4)
                iea5 = utils.iemg(window5)
                iemg_values_channel5.append(iea5)
                iea6 = utils.iemg(window6)
                iemg_values_channel6.append(iea6)
                iea7 = utils.iemg(window7)
                iemg_values_channel7.append(iea7)
                iea8 = utils.iemg(window8)
                iemg_values_channel8.append(iea8)

                rms = [rms_values_channel1,rms_values_channel2,rms_values_channel3,rms_values_channel4,rms_values_channel5,rms_values_channel6,rms_values_channel7,rms_values_channel8]
                wl = [wl_values_channel1,wl_values_channel2,wl_values_channel3,wl_values_channel4,wl_values_channel5,wl_values_channel6,wl_values_channel7,wl_values_channel8]
                ssc = [ssc_values_channel1,ssc_values_channel2,ssc_values_channel3,ssc_values_channel4,ssc_values_channel5,ssc_values_channel6,ssc_values_channel7,ssc_values_channel8]
                zcr = [zcr_values_channel1,zcr_values_channel2,zcr_values_channel3,zcr_values_channel4,zcr_values_channel5,zcr_values_channel6,zcr_values_channel7,zcr_values_channel8]
                mav = [mav_values_channel1,mav_values_channel2,mav_values_channel3,mav_values_channel4,mav_values_channel5,mav_values_channel6,mav_values_channel7,mav_values_channel8]
                ener = [e_values_channel1,e_values_channel2,e_values_channel3,e_values_channel4,e_values_channel5,e_values_channel6,e_values_channel7,e_values_channel8]
                hj = [hj_values_channel1,hj_values_channel2,hj_values_channel3,hj_values_channel4,hj_values_channel5,hj_values_channel6,hj_values_channel7,hj_values_channel8]
                sk = [skew_values_channel1,skew_values_channel2,skew_values_channel3,skew_values_channel4,skew_values_channel5,skew_values_channel6,skew_values_channel7,skew_values_channel8]
                iemg = [iemg_values_channel1,iemg_values_channel2,iemg_values_channel3,iemg_values_channel4,iemg_values_channel5,iemg_values_channel6,iemg_values_channel7,iemg_values_channel8]
                
                features = [zcr,ssc,wl,rms,ener]
                X.append(features)
                Y.append(var)
                rms_values_channel1 = []
                rms_values_channel2 = []
                rms_values_channel3 = []
                rms_values_channel4 = []
                rms_values_channel5 = []
                rms_values_channel6 = []
                rms_values_channel7 = []
                rms_values_channel8 = []
                rms = []
                e_values_channel1 = []
                e_values_channel2 = []
                e_values_channel3 = []
                e_values_channel4 = []
                e_values_channel5 = []
                e_values_channel6 = []
                e_values_channel7 = []
                e_values_channel8 = []
                
                ener = []

                wl_values_channel1 = []
                wl_values_channel2 = []
                wl_values_channel3 = []
                wl_values_channel4 = []
                wl_values_channel5 = []
                wl_values_channel6 = []
                wl_values_channel7 = []
                wl_values_channel8 = []
                wl = []

                ssc_values_channel1 = []
                ssc_values_channel2 = []
                ssc_values_channel3 = []
                ssc_values_channel4 = []
                ssc_values_channel5 = []
                ssc_values_channel6 = []
                ssc_values_channel7 = []
                ssc_values_channel8 = []
                ssc = []

                zcr_values_channel1 = []
                zcr_values_channel2 = []
                zcr_values_channel3 = []
                zcr_values_channel4 = []
                zcr_values_channel5 = []
                zcr_values_channel6 = []
                zcr_values_channel7 = []
                zcr_values_channel8 = []
                zcr = []

                mav_values_channel1 = []
                mav_values_channel2 = []
                mav_values_channel3 = []
                mav_values_channel4 = []
                mav_values_channel5 = []
                mav_values_channel6 = []
                mav_values_channel7 = []
                mav_values_channel8 = []
                mav = []

                hj_values_channel1 = []
                hj_values_channel2 = []
                hj_values_channel3 = []
                hj_values_channel4 = []
                hj_values_channel5 = []
                hj_values_channel6 = []
                hj_values_channel7 = []
                hj_values_channel8 = []
                hj = []

                skew_values_channel1 = []
                skew_values_channel2 = []
                skew_values_channel3 = []
                skew_values_channel4 = []
                skew_values_channel5 = []
                skew_values_channel6 = []
                skew_values_channel7 = []
                skew_values_channel8 = []

                sk = []

                iemg_values_channel1 = []
                iemg_values_channel2 = []
                iemg_values_channel3 = []
                iemg_values_channel4 = []
                iemg_values_channel5 = []
                iemg_values_channel6 = []
                iemg_values_channel7 = []
                iemg_values_channel8 = []

                iemg =[]

                features = []
                

                start += int(self.overlapping*N)

                
                
                

        return X,Y
    



    






    

                
    

    

                
                
                

       
    







    