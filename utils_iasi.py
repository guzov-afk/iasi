import numpy as np


def MAV(data):
    return np.mean(np.abs(data))


def ZCR(data):
	return ((data[:-1] * data[1:]) < 0).sum()


def WL(data):
	return  (abs(data[:-1]-data[1:])).sum()


def RMS(data):
    return np.sqrt(np.mean(data**2))