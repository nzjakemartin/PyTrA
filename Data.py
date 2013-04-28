"""
Class for holding TrA data
"""
import numpy as np


class Data():
	time_C = np.array([])
	wavelength_C = np.array([])
	Chirp = np.array([])
	time = np.array([])
	wavelength = np.array([])
	Three_d = np.array([])
	Three_d_wavelength = np.array([])
	Three_d_time = np.array([])
	TrA_Data = np.array([])
	TrA_Data_gridded = np.array([[0],[0],[0]])
	FFT = np.array([])
	tracefitmodel = []
	Traces = np.array([0])
	Pixels = np.array([])
	Range = np.array([-1,2])
	mcmc = {}