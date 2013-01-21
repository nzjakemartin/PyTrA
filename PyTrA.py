##--PyTrA--##

import numpy as np
from numpy import genfromtxt
from pylab import ginput
import matplotlib.pyplot as plt
import os
from enthought.traits.ui.api import *
from enthought.traits.api import *
from enthought.traits.ui.file_dialog import open_file
import mayavi.mlab as mlab
from scipy import interpolate, special, linalg
from Data import Data
import pymodelfit.fitgui as fitgui
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from pymodelfit import FunctionModel1DAuto
from pymodelfit import register_model
from datetime import date
import scipy.fftpack as fft
from PyTrA_Help import Help

##--Describes exponential models--##

class Convoluted_exp1(FunctionModel1DAuto):
    def f(self,x,T1=1,A1=1,w=1,mu=1,y0=1):
        d = (w/(2*np.sqrt(2*np.log(2))))
        return A1*1/2*np.exp(-x/T1)*np.exp((mu+(d**2)/(2*T1))/T1)*(1+special.erf((x-(mu+(d**2)/T1))/(np.sqrt(2)*d))) + y0

class Convoluted_exp2(FunctionModel1DAuto):
    def f(self,x,T1=1,T2=2,A1=1,A2=1,w=0.25,mu=1,y0=1):
        d = w/(2*np.sqrt(2*np.log(2)))
        return A1*1/2*np.exp(-x/T1)*np.exp((mu+(d**2)/(2*T1))/T1)*(1+special.erf((x-(mu+(d**2)/T1))/(np.sqrt(2)*d))) + A2*1/2*np.exp(-x/T2)*np.exp((mu+(d**2)/(2*T2))/T2)*(1+special.erf((x-(mu+(d**2)/T2))/(np.sqrt(2)*d))) + y0

class Convoluted_exp3(FunctionModel1DAuto):
    def f(self,x,T1=1,T2=1,T3=1,A1=1,A2=1,A3=1,w=1,mu=1,y0=1):
        d = (w/(2*np.sqrt(2*np.log(2))))
        return A1*1/2*np.exp(-x/T1)*np.exp((mu+(d**2)/(2*T1))/T1)*(1+special.erf((x-(mu+(d**2)/T1))/(np.sqrt(2)*d))) + A2*1/2*np.exp(-x/T2)*np.exp((mu+(d**2)/(2*T2))/T2)*(1+special.erf((x-(mu+(d**2)/T2))/(np.sqrt(2)*d))) + A3*1/2*np.exp(-x/T3)*np.exp((mu+(d**2)/(2*T3))/T3)*(1+special.erf((x-(mu+(d**2)/T3))/(np.sqrt(2)*d))) + y0

class Convoluted_exp4(FunctionModel1DAuto):
    def f(self,x,T1=1,T2=1,T3=1,T4=1,A1=1,A2=1,A3=1,A4=1,w=1,mu=1,y0=1,):
        d = (w/(2*np.sqrt(2*np.log(2))))
        return A1*1/2*np.exp(-x/T1)*np.exp((mu+(d**2)/(2*T1))/T1)*(1+special.erf((x-(mu+(d**2)/T1))/(np.sqrt(2)*d))) + A2*1/2*np.exp(-x/T2)*np.exp((mu+(d**2)/(2*T2))/T2)*(1+special.erf((x-(mu+(d**2)/T2))/(np.sqrt(2)*d))) + A3*1/2*np.exp(-x/T3)*np.exp((mu+(d**2)/(2*T3))/T3)*(1+special.erf((x-(mu+(d**2)/T3))/(np.sqrt(2)*d))) + A4*1/2*np.exp(-x/T4)*np.exp((mu+(d**2)/(2*T4))/T4)*(1+special.erf((x-(mu+(d**2)/T4))/(np.sqrt(2)*d))) + y0
		
register_model(Convoluted_exp1, name='Convoluted_exp1', overwrite=False)
register_model(Convoluted_exp2, name='Convoluted_exp2', overwrite=False)
register_model(Convoluted_exp3, name='Convoluted_exp3', overwrite=False)
register_model(Convoluted_exp4, name='Convoluted_exp4', overwrite=False)

##--set up window--##

class OhioLoader(HasTraits):
    Data_file = File()
    Delay = File()
    WaveCal = Array(np.float, (6,2), np.array([[400, 2013.07],[450,1724.29],[500,1397.38],[570,955.733],[600,775.612],[650,458.316]]))
    Load_data = Button("Load data")
    
    view = View(
            Item('Data_file', style = 'simple', label = 'TrA data'),
            Item('Delay', style = 'simple', label = 'Delay'),
            Item('WaveCal', label='Wavelength pixel calibration'),
            Item('Load_data', show_label=False),
            title   = 'Ohio data loader', resizable=True,
            buttons = [ 'OK', 'Cancel' ]
            )
    def _Load_data_fired(self):
        TrA_Raw_T = genfromtxt(self.Data_file, filling_values='0')
    
        # Take transponse of matrix

        Data.TrA_Data = TrA_Raw_T.transpose()
        Data.Chirp = Data.TrA_Data
        TrA_Raw_m, TrA_Raw_n = Data.TrA_Data.shape
        
        Data.time = genfromtxt(self.Delay, filling_values='0')      

        Data.time = Data.time.transpose()

        # Calculating the wavelengths from the calibration data
        
        wave = np.linspace(1, TrA_Raw_n, TrA_Raw_n)
        fitcoeff = np.polyfit(self.WaveCal[:, 1], self.WaveCal[:, 0], 1)

        Data.wavelength = np.polyval(fitcoeff, wave)
        
        #Sort data from smallest wavelength to largest wavelength
        
        inds = Data.wavelength.argsort()
        Data.TrA_Data = Data.TrA_Data[:,inds]
        Data.wavelength = Data.wavelength[inds]
        
        Data.time_C = Data.time
        Data.wavelength_C = Data.wavelength

class FFTfilter(HasTraits):
    raw_plot = Button('2D plot of raw data')
    FFT_raw = Button('FFT')
    filter_high = int(0)
    filter_low = int(0)
    data = np.array([])
    fft_filter = Button('Apply filter')
    accept = Button('Apply to data set')
    
    view = View(
        Item('raw_plot', show_label=False),
        Item('FFT_raw', show_label=False),
        Item('filter_high', show_label=False),
        Item('filter_low', show_label=False),
        Item('fft_filter', show_label=False),
        Item('accept', show_label=False),
        title   = 'FFT filter', resizable=False,
        )
    
    def _raw_plot_fired(self):
        plt.figure()
        plt.contourf(Data.TrA_Data,100,cmap=plt.cm.Greys_r)
        plt.title('Raw data')
        plt.show()
        
    def _FFT_raw_fired(self):
        fft_shift = fft.fft2(Data.TrA_Data)
        Data.FFT = fft.ifftshift(fft_shift)
        plt.figure()
        plt.contourf(np.log(np.abs(Data.FFT)**2),cmap=plt.cm.Greys_r)
        plt.title('FFT of raw')
        plt.show()
    
    def _fft_filter_fired(self):
        Data.FFT[int(self.filter_high):int(self.filter_low),:] = 0
        Data.FFT[int(Data.time.shape[0])-int(self.filter_low)+1:int(Data.time.shape[0])-int(self.filter_high)+1,:] = 0
        plt.figure()
        plt.contourf(np.log(np.abs(Data.FFT)**2),cmap=plt.cm.Greys_r)
        plt.show()
        
        shift = fft.fftshift(Data.FFT)
        self.data = fft.ifft2(shift)
        plt.figure()
        plt.contourf(self.data,200,cmap=plt.cm.Greys_r)
        plt.title('filtered raw data')
        plt.show()
        
    def _accept_fired(self):
        Data.TrA_Data = np.real(self.data)

class MainWindow(HasTraits):
    scene = Instance(MlabSceneModel, ())
    TrA_Raw_file = File("TrA data")
    Chirp_file = File("Chirp data")
    Load_files = Button("Load data")
    Shiftzero = Button("Shift time zero")
    Ohioloader = Button("Ohio data loader")
    DeleteTraces = Button("Delete traces")
    Delete_spectra = Button('Delete spectra')
    fft_filter = Button('FFT filter')
    PlotChirp = Button("2D plot of chirp")
    Timelim = Array(np.float,(1,2))
    Fix_Chirp = Button("Fix for chirp")
    Fit_Trace = Button("Fit trace")
    Fit_Chirp = Button("Fit chirp")
    SVD = Enum(1,2,3,4,5)
    SVD = Button("SVD on plot")
    EFA = Button("Evolving factor analysis")
    Traces_num = 0
    Multiple_Trace = Button("Select multiple traces")
    title = Str("Welcome to PyTrA")
    Plot_3D = Button("3D plot")
    Plot_2D = Button("2D plot")
    Plot_Traces = Button("Plot traces")
    multiple_plots = Button("Multiple traces/spectra on plot")
    Normalise = Button("Normalise")
    Kinetic_Trace = Button("Kinetic trace")
    Spectra = Button("Spectra")
    Trace_Igor = Button("Send traces to Igor")
    Global = Button("Global fit")
    Save_Glo = Button("Save as Glotaran file")
    Save_csv = Button("Save csv with title as file name")
    Clear_Traces = Button("Clear Traces")
    Help = Button("Help")
    Status = Str()
    
    
    view = View(
        VSplit((HSplit((Item('title', show_label=False),
        Item('TrA_Raw_file', style = 'simple', show_label=False),
        Item('Chirp_file', style = 'simple', show_label=False),
        Item('Load_files', show_label=False),
        Item('Ohioloader', show_label=False),
        Item('DeleteTraces', show_label=False),
        Item('Delete_spectra', show_label=False),
        Item('fft_filter', show_label=False),
        Item('Shiftzero', show_label=False),
        Label('Chirp Correction'),
        Item('PlotChirp', show_label=False),
        Label('Time range for chirp corr short/long'),
        Item('Timelim', show_label=False),
        Item('Fix_Chirp', show_label=False),
        Label('Data Analysis'),
        Item('Fit_Trace', show_label=False),
        Item('Plot_2D', show_label=False),
        Item('SVD', show_label=False),
        Item('EFA', show_label=False),
        Label('Global fitting'),
        Item('Multiple_Trace', show_label=False),
        Item('Trace_Igor', show_label=False),
        Item('Plot_Traces', show_label=False),
        Item('Clear_Traces', show_label=False),
        Label('Visualisation'),
        Item('Plot_3D', show_label=False),
        Item('Plot_2D', show_label=False),
        Item('Spectra', show_label=False),
        Item('Kinetic_Trace', show_label=False),
        Item('multiple_plots', show_label=False),
        Item('Normalise', show_label=False),
        Label('Export Data'),
        Item('Save_csv', show_label=False),
        Item('Save_Glo', show_label=False),
        Item('Help', show_label=False)),
        (Item('scene', editor=SceneEditor(scene_class=MayaviScene),height=500, width=700, show_label=False))))),
        (Item('Status', show_label=False)),
        title   = 'PyTrA', resizable=True,
        
        )
    
    def _Load_files_fired(self):
        # Load TrA file into array depends on extension
        Data.filename = self.TrA_Raw_file
        TrA_Raw_file_name, TrA_Raw_file_extension = os.path.splitext(self.TrA_Raw_file)
        TrA_Raw_file_dir, TrA_Raw_file_name = os.path.split(self.TrA_Raw_file)
        TrA_Raw_name, TrA_Raw_ex = os.path.splitext(TrA_Raw_file_name)
        self.title = TrA_Raw_name
        
        if TrA_Raw_file_extension == '.csv':
            TrA_Raw_T = genfromtxt(self.TrA_Raw_file, delimiter=',', filling_values='0')
        elif TrA_Raw_file_extension == '.txt':
            TrA_Raw_T = genfromtxt(self.TrA_Raw_file, delimiter=' ', filling_values='0')

        # Take transponse of matrix

        TrA_Raw = TrA_Raw_T.transpose()

        # Extracts out Data and column values
        
        TrA_Raw_m, TrA_Raw_n = TrA_Raw.shape
        
        Data.time = TrA_Raw[1:TrA_Raw_m, 0]
        Data.wavelength = TrA_Raw[0, 1:TrA_Raw_n]
        Data.TrA_Data = TrA_Raw[1:TrA_Raw_m,1:TrA_Raw_n]
        
        # deleting last time if equal to zero this occurs if data is saved in excel
         
        if Data.TrA_Data[-1,0]==0:
            Data.TrA_Data=Data.TrA_Data[0:-1,:]
            Data.time=Data.time[0:-1]
        
        #Sort data into correct order
        inds = Data.wavelength.argsort()
        Data.TrA_Data = Data.TrA_Data[:,inds]
        Data.wavelength = Data.wavelength[inds]
        
        indst = Data.time.argsort()
        Data.TrA_Data = Data.TrA_Data[indst,:]
        Data.time = Data.time[indst]
        
        # Importing Chirp data
        
        try:
            Chirp_file_name, Chirp_file_extension = os.path.splitext(self.Chirp_file)
            
            if Chirp_file_extension == '.csv':
                Chirp_Raw_T = genfromtxt(self.Chirp_file, delimiter=',', filling_values='0')
            if Chirp_file_extension == '.txt':
                Chirp_Raw_T = genfromtxt(self.Chirp_file, delimiter=' ', filling_values='0')      
    
            Chirp_Raw = Chirp_Raw_T.transpose()
    
            Chirp_Raw_m, Chirp_Raw_n = Chirp_Raw.shape
    
            Data.time_C = Chirp_Raw[1:TrA_Raw_m, 0]
            Data.wavelength_C = Chirp_Raw[0, 1:Chirp_Raw_n]
            Data.Chirp = Chirp_Raw[1:Chirp_Raw_m,1:Chirp_Raw_n]
            
        except:
            self.Status=("No Chirp found: %s"%(self.Status))
        
        self.Status=('Data file imported of size t=%s and wavelength=%s: %s : %s' %(Data.TrA_Data.shape[0],Data.TrA_Data.shape[1],TrA_Raw_name,self.Status))
        
    def _Ohioloader_fired(self):
        ohio = OhioLoader().edit_traits()
        self.Status = ('Data file imported of size %s by %s: %s' %(Data.TrA_Data.shape[0],Data.TrA_Data.shape[1],self.Status))
        
    def _fft_filter_fired(self):
        fft_live = FFTfilter().edit_traits()
        
    def _Shiftzero_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength, Data.time[0:20], Data.TrA_Data[0:20,:], 100)
        plt.title('Pick time zero')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(1))
        plt.show()
        plt.close()

        Data.time = Data.time-fittingto[0][1]
               
    def _DeleteTraces_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick between wavelength to delete (left to right)')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(2))
        plt.show()
        plt.close()
        
        index_wavelength_left=(np.abs(Data.wavelength-fittingto[0,0])).argmin()
        index_wavelength_right=(np.abs(Data.wavelength-fittingto[1,0])).argmin()
        
        if index_wavelength_right <= index_wavelength_left:
            hold = index_wavelength_left
            index_wavelength_left = index_wavelength_right
            index_wavelength_right = hold
        
        if index_wavelength_left == 0:
            Data.TrA_Data = Data.TrA_Data[:,index_wavelength_right:]
            Data.wavelength = Data.wavelength[index_wavelength_right:]
            
        if index_wavelength_right == Data.wavelength.shape:
            Data.TrA_Data = Data.TrA_Data[:,:index_wavelength_left]
            Data.wavelength = Data.wavelength[:index_wavelength_left]
            
        if index_wavelength_left != 0 & index_wavelength_right != Data.wavelength.shape:            
            Data.TrA_Data = np.hstack((Data.TrA_Data[:,:index_wavelength_left],Data.TrA_Data[:,index_wavelength_right:]))
            Data.wavelength = np.hstack((Data.wavelength[:index_wavelength_left],Data.wavelength[index_wavelength_right:]))
            
        self.Status = "Deleted traces between %s and %s: %s" %(fittingto[0,0],fittingto[1,0],self.Status)
        
    def _Delete_spectra_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick between times to delete (top to bottom)')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(2))
        plt.show()
        plt.close()
        
        
        
        index_time_top=(np.abs(Data.time-fittingto[1,1])).argmin()       
        index_time_bottom=(np.abs(Data.time-fittingto[0,1])).argmin()
        
        if index_time_bottom <= index_time_top:
            hold = index_time_top
            index_time_top = index_time_bottom
            index_time_bottom = hold
        
        if index_time_top == 0:
            Data.TrA_Data = Data.TrA_Data[index_time_bottom:,:]
            Data.time = Data.time[index_time_bottom:]
            
        if index_time_bottom == Data.time.shape:
            Data.TrA_Data = Data.TrA_Data[:index_time_top,:]
            Data.time = Data.time[:index_time_top]
            
        if index_time_top != 0 & index_time_bottom != Data.time.shape:
            Data.TrA_Data = np.vstack((Data.TrA_Data[:index_time_top,:],Data.TrA_Data[index_time_bottom:,:]))
            Data.time = np.hstack((Data.time[:index_time_top],Data.time[index_time_bottom:]))
            
        self.Status = "Deleted spectra between %s and %s: %s" %(fittingto[0,1],fittingto[1,1],self.Status)
    
    def _PlotChirp_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength_C, Data.time_C, Data.Chirp, 100)
        plt.title('%s Chirp' %(self.title))
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        plt.show()

    def _Timelim_changed(self):
        Data.Range = self.Timelim
                
    def _Fix_Chirp_fired(self):
        #plot file and pick points for graphing
        plt.figure(figsize=(20,12))
        plt.title('Pick 8 points')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        plt.contourf(Data.wavelength_C, Data.time_C, Data.Chirp, 20)
        plt.ylim((int(Data.Range[0][0]),int(Data.Range[0][1])))
        polypts = np.array(ginput(8))
        plt.show()
        plt.close()
        
        #Fit a polynomial of the form p(x) = p[2] + p[1] + p[0]
        fitcoeff, residuals, rank, singular_values, rcond = np.polyfit(polypts[:, 0], polypts[:, 1], 2, full=True)
        
        stdev = np.sum(residuals**2)/8
        
        #finding where zero time is
        idx=(np.abs(Data.time-0)).argmin()
        plt.figure(figsize=(20,12))
        plt.title("Pick point on wave front")
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        plt.contourf(Data.wavelength, Data.time[idx-1:idx+10], Data.TrA_Data[idx-1:idx+10,:], 100)
        fittingto = np.array(ginput(1)[0])
        plt.show()
        plt.close()
        
        #Moves the chirp inorder to correct coefficient
        fitcoeff[2] = (fitcoeff[0]*fittingto[0]**2 + fitcoeff[1]*fittingto[0] + fittingto[1])*-1
        
        #Iterate over the wavelengths and interpolate for the corrected values
        
        for i in range(0, len(Data.wavelength), 1):
            
            correcttimeval = np.polyval(fitcoeff, Data.wavelength[i])
            f = interpolate.interp1d((Data.time-correcttimeval), (Data.TrA_Data[:, i]), bounds_error=False, fill_value=0)
            fixed_wave = f(Data.time)
            Data.TrA_Data[:, i] = fixed_wave
            
        self.Status = "Polynomial fit with form %s*x^2 + %s*x + %s stdev %s: %s" %(fitcoeff[0],fitcoeff[1],fitcoeff[2],stdev,self.Status)
        
    def _Fit_Trace_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick wavelength to fit')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(1))
        plt.show()
        plt.close()
        
        index_wavelength=(np.abs(Data.wavelength-fittingto[:,0])).argmin()
        Data.tracefitmodel = fitgui.fit_data(Data.time,Data.TrA_Data[:,index_wavelength],autoupdate=False,model=Convoluted_exp1,include_models='Convoluted_exp1,Convoluted_exp2,Convoluted_exp3,Convoluted_exp4')
        
        #If you want to have the fitting gui in another window while PyTrA remains responsive change the fit model to a model instance and use the line bellow to call it
        #Data.tracefitmodel.edit_traits()
        
        self.Status= ('Fitted parameters at wavelength %s, %s: %s'%(fittingto[:,0], Data.tracefitmodel.pardict,self.Status))
        
    def _SVD_fired(self):
        
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        
        index_wavelength_left=(np.abs(Data.wavelength-xmin)).argmin()
        index_wavelength_right=(np.abs(Data.wavelength-xmax)).argmin()
        
        index_time_left=(np.abs(Data.time-ymin)).argmin()
        index_time_right=(np.abs(Data.time-ymax)).argmin()
        
        U, s, V_T = linalg.svd(Data.TrA_Data[index_time_left:index_time_right,index_wavelength_left:index_wavelength_right])
        
        f=plt.figure()
        f.text(0.5,0.975,("SVD %s" %(self.title)),horizontalalignment='center',verticalalignment='top')
        plt.subplot(341)
        plt.plot(Data.time[index_time_left:index_time_right],U[:,0])
        plt.title("1")
        plt.xlabel("time (ps)")
        plt.ylabel("abs.")
        plt.subplot(342)
        plt.plot(Data.time[index_time_left:index_time_right],U[:,1])
        plt.title("2")
        plt.xlabel("time (ps)")
        plt.ylabel("abs.")
        plt.subplot(343)
        plt.plot(Data.time[index_time_left:index_time_right],U[:,2])
        plt.title("3")
        plt.xlabel("time (ps)")
        plt.ylabel("abs.")
        plt.subplot(344)
        plt.plot(Data.time[index_time_left:index_time_right],U[:,3])
        plt.title("4")
        plt.xlabel("time (ps)")
        plt.ylabel("abs.")
        plt.subplot(345)
        plt.plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[0,:])
        plt.title("%s" %(s[0]))
        plt.xlabel("wavelength (nm)")
        plt.ylabel("abs.")
        plt.subplot(346)
        plt.plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[1,:])
        plt.title("%s" %(s[1]))
        plt.xlabel("wavelength (nm)")
        plt.ylabel("abs.")
        plt.subplot(347)
        plt.plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[2,:])
        plt.title("%s" %(s[2]))
        plt.xlabel("wavelength (nm)")
        plt.ylabel("abs.")
        plt.subplot(348)
        plt.plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[3,:])
        plt.title("%s" %(s[3]))
        plt.xlabel("wavelength (nm)")
        plt.ylabel("abs.")
        plt.subplot(349)
        [SVD_1_x,SVD_1_y]=np.meshgrid(V_T[0,:],U[:,0])
        SVD_1 = np.multiply(SVD_1_x,SVD_1_y)*s[0]
        plt.contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_1,50)
        plt.subplot(3,4,10)
        [SVD_2_x,SVD_2_y]=np.meshgrid(V_T[1,:],U[:,1])
        SVD_2 = np.multiply(SVD_2_x,SVD_2_y)*s[1]
        plt.contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_2,50)
        plt.subplot(3,4,11)
        [SVD_3_x,SVD_3_y]=np.meshgrid(V_T[2,:],U[:,2])
        SVD_3 = np.multiply(SVD_3_x,SVD_3_y)*s[2]
        plt.contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_3,50)
        plt.subplot(3,4,12)
        [SVD_4_x,SVD_4_y]=np.meshgrid(V_T[3,:],U[:,3])
        SVD_4 = np.multiply(SVD_4_x,SVD_4_y)*s[3]
        plt.contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_4,50)
        plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.94, wspace=0.2, hspace=0.2)
        plt.show()
        
        plt.figure()
        plt.semilogy(s[0:9],'*')
        plt.title("First 10 singular values")
        plt.show()
       
        self.Status = "First 5 singular values %s in range wavelength %s to %s, time %s to %s : %s" %(s[0:5], xmin, xmax, ymin, ymax, self.Status)
        
    def _EFA_fired(self):
        
        #number of singular values to track
        singvals = 3
        
        #Time
        rows = Data.TrA_Data.shape[0]
        forward_r = np.zeros((rows,singvals))
        backward_r = np.zeros((rows,singvals))
        
        stepl_r = rows-singvals
        #Forward
        
        #Must start with number of tracked singular values in order to intially generate 10 SV
        for i in range(singvals,rows):
            partsvd = linalg.svdvals(Data.TrA_Data[:i,:]).T
            forward_r[i,:] = partsvd[:singvals]
        
        #Backwards
        
        for i in range(0,stepl_r):
            j = (rows-singvals)-i
            partsvd = linalg.svdvals(Data.TrA_Data[j:,:]).T
            backward_r[j,:] = partsvd[:singvals]
        
        plt.figure()
        plt.semilogy(Data.time[singvals:],forward_r[singvals:,:],'b',Data.time[:(rows-singvals)],backward_r[:(rows-singvals),:],'r')
        plt.title("%s EFA time" %(self.title))
        plt.xlabel("Time (ps)")
        plt.ylabel("Log(EV)")
        plt.show()
        
        #Wavelength
        
        cols = Data.TrA_Data.shape[1]
        forward_c = np.zeros((cols,singvals))
        backward_c = np.zeros((cols,singvals))
        
        stepl_c = cols-singvals
        #Forward
        
        #Must start with number of tracked singular values in order to intially generate 10 SV
        for i in range(singvals,cols):
            partsvd = linalg.svdvals(Data.TrA_Data[:,:i])
            forward_c[i,:] = partsvd[:singvals]
        
        #Backwards
        
        for i in range(0,stepl_c):
            j = (cols-singvals)-i
            partsvd = linalg.svdvals(Data.TrA_Data[:,j:])
            backward_c[j,:] = partsvd[:singvals]
        
        plt.figure()
        plt.semilogy(Data.wavelength[singvals:],forward_c[singvals:,:],'b',Data.wavelength[:cols-singvals],backward_c[:cols-singvals,:],'r')
        plt.title("%s EFA wavelength" %(self.title))
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Log(EV)")
        plt.show()
               
    def _Multiple_Trace_fired(self):
        
        plt.figure(figsize=(15,10))
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick between wavelength to fit (left to right)')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(2))
        plt.show()
        plt.close()
        
        index_wavelength_left=(np.abs(Data.wavelength-fittingto[0,0])).argmin()
        index_wavelength_right=(np.abs(Data.wavelength-fittingto[1,0])).argmin()

        Data.Traces = Data.TrA_Data[:,index_wavelength_left:index_wavelength_right].transpose()
        
        self.Status= '%s Traces saved from %s to %s: %s' %(Data.Traces.shape[0], fittingto[0,0], fittingto[1,0],self.Status)

    def _Plot_3D_fired(self):

		xmin, xmax = plt.xlim()
		ymin, ymax = plt.ylim()
		
		index_wavelength_left=(np.abs(Data.wavelength-xmin)).argmin()
		index_wavelength_right=(np.abs(Data.wavelength-xmax)).argmin()
		
		index_time_left=(np.abs(Data.time-ymin)).argmin()
		index_time_right=(np.abs(Data.time-ymax)).argmin()
		
		Data.Three_d = Data.TrA_Data[index_time_left:index_time_right,index_wavelength_left:index_wavelength_right]
		Data.Three_d_wavelength = Data.wavelength[index_wavelength_left:index_wavelength_right]
		Data.Three_d_time = Data.time[index_time_left:index_time_right]
		
		self.scene.mlab.clf()
		
		x = np.linspace(Data.Three_d_wavelength[0],Data.Three_d_wavelength[-1],len(Data.Three_d_wavelength))
		y = np.linspace(Data.Three_d_time[0], Data.Three_d_time[-1],len(Data.Three_d_wavelength))
		[xi,yi] = np.meshgrid(x,y)

		for i in range(len(Data.Three_d_wavelength)):
			repeating_wavelength = np.array(np.ones((len(Data.Three_d_time)))*Data.Three_d_wavelength[i])
			vectors = np.array([Data.Three_d_time,repeating_wavelength,Data.Three_d[:,i]])
			if i==0:
				Data.TrA_Data_gridded = vectors
			else:
				Data.TrA_Data_gridded = np.hstack((Data.TrA_Data_gridded, vectors))

		zi = interpolate.griddata((Data.TrA_Data_gridded[1,:],Data.TrA_Data_gridded[0,:]),Data.TrA_Data_gridded[2,:],(xi,yi), method='linear', fill_value=0)
        
        #Sends 3D plot to mayavi in gui
        
		self.plot = self.scene.mlab.surf(yi,xi,zi, warp_scale=-np.max(Data.Three_d)*100000)
		self.scene.mlab.colorbar(orientation="vertical")
		self.scene.mlab.axes(nb_labels=5)
		self.scene.mlab.ylabel("wavelength (nm)")
		self.scene.mlab.xlabel("time (ps)")
		
    def _Plot_2D_fired(self):
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 200)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Times (ps)')
        plt.title(self.title)
        plt.colorbar()
        plt.show()
        
    def _Plot_Traces_fired(self):
        plt.figure(figsize=(15,10))
        plt.plot(Data.time, Data.Traces.transpose())
        plt.title("%s Traces" %(self.title))
        plt.xlabel('Time')
        plt.ylabel('Abs')
        plt.show()
        
    def _Kinetic_Trace_fired(self):
        
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick wavelength')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(1))
        plt.show()
        plt.close()
        
        index_wavelength=(np.abs(Data.wavelength-fittingto[:,0])).argmin()
        
        plt.figure(figsize=(20,12))
        plt.plot(Data.time, Data.TrA_Data[:,index_wavelength])
        plt.title("%s %s" %(self.title, Data.wavelength[index_wavelength]))
        plt.xlabel('Time')
        plt.ylabel('Abs')
        plt.show()
        
    def _Spectra_fired(self):
        
        plt.figure()
        plt.contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
        plt.title('Pick time')
        plt.xlabel('Wavelength')
        plt.ylabel('Time')
        fittingto = np.array(ginput(1))
        plt.show()
        plt.close()
        
        index_time=(np.abs(Data.time-fittingto[:,1])).argmin()
        
        plt.figure()
        plt.plot(Data.wavelength, Data.TrA_Data[index_time,:])
        plt.title("%s %s" %(self.title, Data.time[index_time]))
        plt.xlabel('Wavelength')
        plt.ylabel('Abs')
        plt.show()
        
    def _multiple_plots_fired(self):
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        
        index_wavelength_left=(np.abs(Data.wavelength-xmin)).argmin()
        index_wavelength_right=(np.abs(Data.wavelength-xmax)).argmin()
        
        index_time_left=(np.abs(Data.time-ymin)).argmin()
        index_time_right=(np.abs(Data.time-ymax)).argmin()
        
        indexwave = int((index_wavelength_right-index_wavelength_left)/10)
        
        # spectrum from every 10th spectra
        
        timevec = np.ones([Data.time[index_time_left:index_time_right].shape[0],10])
        time = np.ones([Data.time[index_time_left:index_time_right].shape[0],10])
        wavelengthvals = np.ones(10)
        
        for i in range(10):
            timevec[:,i] = np.average(Data.TrA_Data[index_time_left:index_time_right,index_wavelength_left+((i)*indexwave):index_wavelength_left+((i)*indexwave)+indexwave],axis=1)
            time[:,i] = Data.time[index_time_left:index_time_right]
            wavelengthvals[i] = round(np.average(Data.wavelength[index_wavelength_left+((i)*indexwave):index_wavelength_left+((i)*indexwave)+indexwave]),1)

        plt.figure()
        colormap = plt.cm.jet
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 10)])
        plt.plot(time,timevec)
        plt.legend(wavelengthvals)
        plt.xlabel('Time (ps)')
        plt.ylabel('Abs.')
        plt.title("Averaged %s %s" %(self.title, 'Wavelengths (nm)'))
        plt.show()
        
        indextime = int((index_time_right-index_time_left)/10)
        
        wavevec = np.ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
        wave = np.ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
        timevals = np.ones(10)
        
        for i in range(10):
            wavevec[:,i] = np.average(Data.TrA_Data[index_time_left+((i)*indextime):index_time_left+((i)*indextime)+indextime,index_wavelength_left:index_wavelength_right],axis=0)
            wave[:,i] = Data.wavelength[index_wavelength_left:index_wavelength_right]
            timevals[i] = round(np.average(Data.time[index_time_left+((i)*indextime):index_time_left+((i)*indextime)+indextime]),1)

        plt.figure()
        colormap = plt.cm.jet
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 10)])
        plt.plot(wave,wavevec)
        plt.legend(timevals)
        plt.title("Averaged %s %s" %(self.title, 'Times (ps)'))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Abs.')
        plt.show()
        
    def _Normalise_fired(self):
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        
        index_wavelength_left=(np.abs(Data.wavelength-xmin)).argmin()
        index_wavelength_right=(np.abs(Data.wavelength-xmax)).argmin()
        
        index_time_left=(np.abs(Data.time-ymin)).argmin()
        index_time_right=(np.abs(Data.time-ymax)).argmin()
        
        indextime = int((index_time_right-index_time_left)/10)
        
        wavevec = np.ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
        wave = np.ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
        timevals = np.ones(10)
        
        for i in range(10):
            wavevec[:,i] = Data.TrA_Data[(index_time_left+((i)*indextime)),index_wavelength_left:index_wavelength_right]
            max_i = np.max(wavevec[:,i])
            min_i = np.min(wavevec[:,i])
            wavevec[:,i] = (wavevec[:,i]-min_i)/(max_i-min_i)
            wave[:,i] = Data.wavelength[index_wavelength_left:index_wavelength_right]
            timevals[i] = Data.time[index_time_left+((i)*indextime)]

        plt.figure()
        colormap = plt.cm.jet
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 10)])
        plt.plot(wave,wavevec)
        plt.jet()
        plt.legend(timevals)
        plt.title("Normalised %s %s" %(self.title, 'Times (ps)'))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Abs.')
        plt.show()        
    
        indexwave = int((index_wavelength_right-index_wavelength_left)/10)
        
        # spectrum from every 10th spectra
        
        timevec = np.ones([Data.time[index_time_left:index_time_right].shape[0],10])
        time = np.ones([Data.time[index_time_left:index_time_right].shape[0],10])
        wavelengthvals = np.ones(10)
        
        for i in range(10):
            timevec[:,i] = Data.TrA_Data[index_time_left:index_time_right,(index_wavelength_left+((i)*indexwave))]
            max2_i = np.max(timevec[:,i])
            min2_i = np.min(timevec[:,i])
            timevec[:,i] = (timevec[:,i]-min2_i)/(max2_i-min2_i)                              
            time[:,i] = Data.time[index_time_left:index_time_right]
            wavelengthvals[i] = Data.wavelength[index_wavelength_left+((i)*indexwave)]

        plt.figure()
        colormap = plt.cm.jet
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 10)])
        plt.plot(time,timevec)
        plt.legend(wavelengthvals)
        plt.xlabel('Time (ps)')
        plt.ylabel('Abs.')
        plt.title("Normalised %s %s" %(self.title, 'Wavelengths (nm)'))
        plt.show()
    	
    def _Trace_Igor_fired(self):
        
        try:
            import win32com.client # Communicates with Igor needs pywin32 library
            f=open(("%s\Traces.txt" %(os.path.dirname(self.TrA_Raw_file))), 'w')
            for i in range(len(Data.time)):
                f.write("%s" %(Data.time[i]))
                for j in range(len(Data.Traces)):
                    f.write(",%s" %(Data.Traces[j,i]))
                f.write("\n")
            f.close()
            
            # Sends traces to Igor and opens up Global fitting gui in Igor
            igor=win32com.client.Dispatch("IgorPro.Application")
        
            #Load into igor using LoadWave(/A=Traces/J/P=pathname) /J specifies it as a txt delimited file
            igor.Execute('NewPath pathName, "%s"' %(os.path.dirname(self.TrA_Raw_file)))
            igor.Execute('Loadwave/J/P=pathName "Traces.txt"')
            igor.Execute('Rename wave0,timeval')
        
            # Run global fitting gui in Igor
            igor.Execute('WM_NewGlobalFit1#InitNewGlobalFitPanel()')
            igor.clear()
            
        except:
            self.Status = 'setuptools not installed or Igor not open. Saved traces into directory: %s' %(self.Status)
            try:
                f=open(("%s\Traces.txt" %(os.path.dirname(self.TrA_Raw_file))), 'w')
                for i in range(len(Data.time)):
                    f.write("%s" %(Data.time[i]))
                    for j in range(len(Data.Traces)):
                        f.write(",%s" %(Data.Traces[j,i]))
                    f.write("\n")
                f.close()
            except:
                self.Status = 'Please select multiple traces: %s' %(self.Status)

    def _Save_Glo_fired(self):
        # Generates ouput file in Glotaran Time explicit format
        pathname = "%s\Glotaran.txt" %(os.path.dirname(self.TrA_Raw_file))
        f = open(pathname, 'w')
        f.write("#-#-#-#-#-# Made with PyTrA #-#-#-#-#-#\n")
        f.write("\n")
        f.write("Time explicit\n")
        f.write("intervalnr %d\n" %(len(Data.time)))
        for i in range(len(Data.time)):
            f.write(" %s" %(Data.time[i]))
        f.write("\n")
        for i in range(len(Data.wavelength)):
            f.write("%s" %(Data.wavelength[i]))
            for j in range(len(Data.time)):
                f.write(" %s" %(Data.TrA_Data[j,i]))
            f.write("\n")
        
        self.Status = 'Saved Glotaran file to TrA data file directory: %s' %(self.Status)
        
    def _Save_csv_fired(self):
        now = date.today()
        pathname = "%s\Saved%s%s.csv" %(os.path.dirname(self.TrA_Raw_file), now.strftime("%m-%d-%y"),self.title)
        f = open(pathname, 'w')
        f.write("0")
        for i in range(len(Data.time)):
            f.write(",%s" %(Data.time[i]))
        f.write("\n")
        for i in range(len(Data.wavelength)):
            f.write("%s" %(Data.wavelength[i]))
            for j in range(len(Data.time)):
                f.write(",%s" %(Data.TrA_Data[j,i]))
            f.write("\n")
        
        self.Status= 'Saved to TrA data file directory: %s' %(self.Status)
        
    def _Clear_Traces_fired(self):
        self.Traces_num = 0
        Data.Traces = 0
        self.Status='Cleared all of the traces: %s' %(self.Status)
        
    def _Help_fired(self):
        help = Help().edit_traits()

if __name__=='__main__':
        main = MainWindow().configure_traits()
