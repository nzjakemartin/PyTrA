'''
PyTrA: Python based fitting of femtosecond transient absorption spectroscopy data
Author: Jake Martin, Photon Factory, University of Auckland
'''

import numpy as np
from datetime import date
from os import startfile
from os.path import dirname,splitext,split
from numpy import sqrt, log, exp, array, linspace, polyfit, polyval, abs, hstack, vstack, sum, meshgrid, multiply, zeros, arange, ones, average, delete
from numpy import genfromtxt
from pylab import ginput
from scipy import interpolate, special, linalg
from matplotlib.pyplot import figure, contourf, title, xlabel, ylabel, show, close, xlim, ylim, subplot, semilogy, gca, colorbar, cm, legend, plot, subplots_adjust
from traits.api import  HasTraits, File, Button, Array, Instance, Str, Int, Float, Bool
from traitsui.menu import Action, Menu,MenuBar
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, VSplit, RangeEditor, ValueEditor
from chaco.api import ArrayPlotData, Plot, jet
from chaco.tools.api import PanTool, ZoomTool
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from enable.component_editor import ComponentEditor
from pymodelfit import fitgui, FunctionModel1DAuto, register_model
from Data import Data
import mcmc
from PyTrA_Help import Help

##--Describes exponential models--##

class Convoluted_exp1(FunctionModel1DAuto):
	def f(self,x,Tone=20,Aone=0.1,fwhm=0.5,mu=0,c=0):
		d = (fwhm/(2*sqrt(2*log(2))))
		return Aone*1/2*exp(-x/Tone)*exp((mu+(d**2)/(2*Tone))/Tone)*(1+special.erf((x-(mu+(d**2)/Tone))/(sqrt(2)*d))) + c

class Convoluted_exp2(FunctionModel1DAuto):
	def f(self,x,Tone=20,Ttwo=20,Aone=0.1,Atwo=0.1,fwhm=0.5,mu=0,c=0):
		d = fwhm/(2*sqrt(2*log(2)))
		return Aone*1/2*exp(-x/Tone)*exp((mu+(d**2)/(2*Tone))/Tone)*(1+special.erf((x-(mu+(d**2)/Tone))/(sqrt(2)*d))) + Atwo*1/2*exp(-x/Ttwo)*exp((mu+(d**2)/(2*Ttwo))/Ttwo)*(1+special.erf((x-(mu+(d**2)/Ttwo))/(sqrt(2)*d))) + c

class Convoluted_exp3(FunctionModel1DAuto):
	def f(self,x,Tone=20,Ttwo=20,Tthree=20,Aone=0.1,Atwo=0.1,Athree=0.1,fwhm=0.5,mu=0,c=0):
		d = (fwhm/(2*sqrt(2*log(2))))
		return Aone*1/2*exp(-x/Tone)*exp((mu+(d**2)/(2*Tone))/Tone)*(1+special.erf((x-(mu+(d**2)/Tone))/(sqrt(2)*d))) + Atwo*1/2*exp(-x/Ttwo)*exp((mu+(d**2)/(2*Ttwo))/Ttwo)*(1+special.erf((x-(mu+(d**2)/Ttwo))/(sqrt(2)*d))) + Athree*1/2*exp(-x/Tthree)*exp((mu+(d**2)/(2*Tthree))/Tthree)*(1+special.erf((x-(mu+(d**2)/Tthree))/(sqrt(2)*d))) + c

class Convoluted_exp4(FunctionModel1DAuto):
	def f(self,x,Tone=20,Ttwo=20,Tthree=20,Tfour=20,Aone=0.1,Atwo=0.1,Athree=0.1,Afour=0.1,fwhm=0.5,mu=0,c=0):
		d = (fwhm/(2*sqrt(2*log(2))))
		return Aone*1/2*exp(-x/Tone)*exp((mu+(d**2)/(2*Tone))/Tone)*(1+special.erf((x-(mu+(d**2)/Tone))/(sqrt(2)*d))) + Atwo*1/2*exp(-x/Ttwo)*exp((mu+(d**2)/(2*Ttwo))/Ttwo)*(1+special.erf((x-(mu+(d**2)/Ttwo))/(sqrt(2)*d))) + Athree*1/2*exp(-x/Tthree)*exp((mu+(d**2)/(2*Tthree))/Tthree)*(1+special.erf((x-(mu+(d**2)/Tthree))/(sqrt(2)*d))) + Afour*1/2*exp(-x/Tfour)*exp((mu+(d**2)/(2*Tfour))/Tfour)*(1+special.erf((x-(mu+(d**2)/Tfour))/(sqrt(2)*d))) + c

class Gauss_1(FunctionModel1DAuto):
	def f(self,x,Aone=0.1,fwhm=30.0,mu=500.0,c=0.0):
		d = (fwhm/(2*sqrt(2*log(2))))
		return Aone*exp(-((x-mu)**2)/(2*d**2))+c

class Gauss_2(FunctionModel1DAuto):
	def f(self,x,Aone=0.1,Atwo=0.1,fwhm_1=40.0,fwhm_2=30.0,mu_1=500,mu_2=600,c=0.0):
		d_1 = (fwhm_1/(2*sqrt(2*log(2))))
		d_2 = (fwhm_2/(2*sqrt(2*log(2))))
		return Aone*exp(-((x-mu_1)**2)/(2*d_1**2))+Atwo*exp(-((x-mu_2)**2)/(2*d_2**2))+c

class Gauss_3(FunctionModel1DAuto):
	def f(self,x,Aone=0.1,Atwo=0.1,Athree=0.2,fwhm_1=40.0,fwhm_2=30.0,fwhm_3=20,mu_1=500,mu_2=600,mu_3=700,c=0.0):
		d_1 = (fwhm_1/(2*sqrt(2*log(2))))
		d_2 = (fwhm_2/(2*sqrt(2*log(2))))
		d_3 = (fwhm_3/(2*sqrt(2*log(2))))
		return Aone*exp(-((x-mu_1)**2)/(2*d_1**2))+Atwo*exp(-((x-mu_2)**2)/(2*d_2**2))+Athree*exp(-((x-mu_3)**2)/(2*d_3**2))+c

class Gauss_4(FunctionModel1DAuto):
	def f(self,x,Aone=0.1,Atwo=0.1,Athree=0.2,Afour=0.2,fwhm_1=40.0,fwhm_2=30.0,fwhm_3=20,fwhm_4=20,mu_1=500,mu_2=600,mu_3=700,mu_4=750,c=0.0):
		d_1 = (fwhm_1/(2*sqrt(2*log(2))))
		d_2 = (fwhm_2/(2*sqrt(2*log(2))))
		d_3 = (fwhm_3/(2*sqrt(2*log(2))))
		d_4 = (fwhm_4/(2*sqrt(2*log(2))))
		return Aone*exp(-((x-mu_1)**2)/(2*d_1**2))+Atwo*exp(-((x-mu_2)**2)/(2*d_2**2))+Athree*exp(-((x-mu_3)**2)/(2*d_3**2))+Afour*exp(-((x-mu_4)**2)/(2*d_4**2))+c

register_model(Convoluted_exp1, name='Convoluted_exp1', overwrite=False)
register_model(Convoluted_exp2, name='Convoluted_exp2', overwrite=False)
register_model(Convoluted_exp3, name='Convoluted_exp3', overwrite=False)
register_model(Convoluted_exp4, name='Convoluted_exp4', overwrite=False)
register_model(Gauss_1, name='Gauss_1', overwrite=False)
register_model(Gauss_2, name='Gauss_2', overwrite=False)
register_model(Gauss_3, name='Gauss_3', overwrite=False)
register_model(Gauss_4, name='Gauss_4', overwrite=False)

class OhioLoader(HasTraits):
	'''
	Window to import data from Ohio State University Ultrafast TrA system. Calibration of wavelength to pixel is done and imported into a file that PyTrA can use.
	'''
	Data_file = File("TrA data")
	Delay = File("Delay file")
	WaveCal = Array(float, (6,2), array([[400, 2013.07],[450,1724.29],[500,1397.38],[570,955.733],[600,775.612],[650,458.316]]))
	Load_data = Button("Load data")

	view = View(
			Label('Importer for Ohio State University ultrafast TrA system'),
			Item('Data_file', style = 'simple', show_label=False),
			Item('Delay', style = 'simple', show_label=False),
			Label('Wavelength pixel calibration'),
			Item('WaveCal', show_label=False),
			Item('Load_data', show_label=False),
			title   = 'Ohio data loader', resizable=True,
			buttons = [ 'OK', 'Cancel' ]
			)
	def _Load_data_fired(self):
		'''
		Data is loaded in from file and calibrated to the correct wavelength values
		'''
		TrA_Raw_T = genfromtxt(self.Data_file, filling_values='0')

		# Take transponse of matrix

		Data.TrA_Data = TrA_Raw_T.transpose()
		Data.Chirp = Data.TrA_Data
		TrA_Raw_m, TrA_Raw_n = Data.TrA_Data.shape

		Data.time = genfromtxt(self.Delay, filling_values='0')

		Data.time = Data.time.transpose()

		# Calculating the wavelengths from the calibration data

		wave = linspace(1, TrA_Raw_n, TrA_Raw_n)
		fitcoeff = polyfit(self.WaveCal[:, 1], self.WaveCal[:, 0], 1)

		Data.wavelength = polyval(fitcoeff, wave)

		#Sort data from smallest wavelength to largest wavelength

		inds = Data.wavelength.argsort()
		Data.TrA_Data = Data.TrA_Data[:,inds]
		Data.wavelength = Data.wavelength[inds]

		Data.time_C = Data.time
		Data.wavelength_C = Data.wavelength

class Dataviewer(HasTraits):
	data = Data

	view = View(
		Item('data',show_label=False,editor=ValueEditor()),
		resizable=True
	)

class MainWindow(HasTraits):
	'''
	Main window for PyTrA sets up the view objects and views the different methods that can be applied to the data
	'''

	#--View objects for TraitsUI--#

	scene = Instance(MlabSceneModel, ())
	plot2D = Instance(Plot)
	plotwaveindexshow = Float
	plotwaveindex = Float
	wavelow=Float(0)
	wavehigh=Float(100)
	waverange=int(100)
	plottimeindexshow = Float
	plottimeindex = Float
	timelow=Float(-1.0)
	timehigh=Float(100.0)
	timerange=int(100)
	plottime = Instance(Plot)
	updateplots = Button('Reset plots')
	zoom_on = Bool(False)
	plotwavelength = Instance(Plot)
	plot2Db = Button("plot")
	TrA_Raw_file = File("TrA data")
	Chirp_file = File("Chirp data")
	Load_files = Button("Load data")
	open_csv = Button("Open csv")
	Shiftzero = Button("Shift time zero")
	Ohioloader = Button("Ohio data loader")
	DeleteTraces = Button("Delete multiple traces")
	Delete_spectra = Button('Delete multiple spectra')
	DeleteTraces_1 = Button('Delete trace')
	DeleteSpectra_1 = Button('Delete spectrum')
	PlotChirp = Button("2D plot of chirp")
	Timelim = Array(float,(1,2))
	Fix_Chirp = Button("Fix for chirp")
	Fit_Trace = Button("Fit Trace")
	Fit_Spec = Button("Fit Spectra")
	mcmc = Button("MCMC")
	Fit_Chirp = Button("Fit chirp")
	SVD = Button("SVD")
	EFA = Button("EFA")
	Traces_num = 0
	Multiple_Trace = Button("Select multiple traces")
	Global = Button("Global fit")
	savetraces = Button("save traces")
	title = Str("Welcome to PyTrA")
	z_height = Int(4)
	Plot_3D = Button("3D plot")
	Plot_2D = Button("2D plot")
	Plot_Traces = Button("Plot traces")
	multiple_plots = Button("Averaged")
	Normalise = Button("Normalised")
	Kinetic_Trace = Button("Kinetic trace")
	Spectra = Button("Spectra")
	Save_Glo = Button("Save as Glotaran file")
	Save_csv = Button("Save csv with title as file name")
	Save_log = Button("Save log file")
	Help = Button("Help")
	log = Str("PyTrA:Python based fitting of Ultra-fast Transient Absorption Data")
	showdata=Button("Show python data")

	#--Setting up views in TraitsUI--#

	threed_group = Group(
		HSplit(Item('Plot_3D', show_label=False),
			Item('z_height', show_label=False)
		),
		Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
		label='3d graph'
	)

	log_group = Group(
		Item('log',style='custom',show_label=False),
		label='log file'
	)

	plot_group = Group(
		HSplit(
		VSplit(
			(
			Label('Hold Ctrl to zoom in plots'),
			Item('plot2D', editor=ComponentEditor(), show_label=False),
			(HSplit(Item('updateplots', show_label=False),Item('zoom_on'),Item('Plot_2D', show_label=False),Item('multiple_plots', show_label=False),Item('Normalise', show_label=False),Item('SVD', show_label=False),Item('EFA', show_label=False))
			)
			),
			(Item('log',style='custom',show_label=False),
			HSplit(Item('Save_log', show_label=False))),
		),
		VSplit(
			(
			Item('plottime', editor=ComponentEditor(), show_label=False),
			(HSplit(Item('Kinetic_Trace', show_label=False),Item('Fit_Trace', show_label=False),Item('mcmc',show_label=False),Item('DeleteTraces_1', show_label=False),Item('plotwaveindexshow', show_label=False, style='readonly'))),
			Item('plotwaveindex', show_label=False, editor=RangeEditor(low_name='wavelow',high_name='wavehigh',format='%6f',label_width=waverange,mode='slider'))
			),
			(
			Item('plotwavelength', editor=ComponentEditor(), show_label=False),
			(HSplit(Item('Spectra', show_label=False),Item('Fit_Spec', show_label=False),Item('DeleteSpectra_1', show_label=False),Item('plottimeindexshow',show_label=False,style='readonly'))),
			Item('plottimeindex', show_label=False, editor=RangeEditor(low_name='timelow',high_name='timehigh',format='%d',label_width=timerange,mode='slider'))
			),
		),
		),
		label='2D plot'
	)

	view = View(

		Group(HSplit(
		Item('TrA_Raw_file', style = 'simple', show_label=False),
		Item('Chirp_file', style = 'simple', show_label=False),
		Item('Load_files', show_label=False),
		Item('title', show_label=False)),
		Tabbed(
			plot_group,
			threed_group
			)),
		menubar=MenuBar(Menu(
						Action(name="Ohio data loader",action="_Ohioloader_fired"),
						Action(name="Spreadsheet",action="_open_csv_fired"),
						Action(name="Save csv",action="_Save_csv_fired"),
						Action(name="Save as Glotaran",action="_Save_Glo_fired"),
						Action(name="Delete multple traces",action="_DeleteTraces_fired"),
						Action(name="Delete multple spectra",action="_Delete_spectra_fired"),
						Action(name="Show python data",action='_showdata_fired'),
						name="File"),
						Menu(
						Action(name="Plot chirp",action='_PlotChirp_fired'),
						Action(name="Chirp correction",action='_Fix_Chirp_fired'),
						Action(name="Shift time zero",action='_Shiftzero_fired'),
						name="Chirp correction"
						),
						Menu(
						Action(name="Select traces",action='_Multiple_Trace_fired'),
						Action(name="Plot traces",action='_Plot_Traces_fired'),
						Action(name='Save traces to File',action='_savetraces_fired'),
						Action(name='Send to Igor',action='_Trace_Igor_fired'),
						name="Global fit"
						),
						Menu(
						Action(name="Help file", action='_Help_fired'),
						name="Help"
						)
						),
		title   = 'PyTrA', resizable=True,

		)

	def _plot2Db_fired(self):
		'''
		Plotting the 2D chaco plot
		'''

		#self.grid_data()

		ds = ArrayPlotData()
		ds.set_data('img',Data.TrA_Data)

		img = Plot(ds)
		cmapImgPlot = img.img_plot("img",colormap=jet,xbounds=(Data.wavelength[0],Data.wavelength[-1]),ybounds=(0,(len(Data.time)-1)))

		self.plot2D = img

		self.plot2D.x_axis.title="Wavelength (nm)"
		self.plot2D.y_axis.title="Samples"

		if self.zoom_on==True:
			zoom = ZoomTool(component=img, tool_mode="box", always_on=True)
			img.overlays.append(zoom)

		img.tools.append(PanTool(img))

		self._plottimeindex_changed()
		self._plotwaveindex_changed()

	def _zoom_on_changed(self):
		self._updateplots_fired()

	def _updateplots_fired(self):
		#Don't regrid just reset 2D and 1D graphs
		ds = ArrayPlotData()
		ds.set_data('img',Data.TrA_Data)

		img = Plot(ds)
		cmapImgPlot = img.img_plot("img",colormap=jet,xbounds=(Data.wavelength[0],Data.wavelength[-1]),ybounds=(0,(len(Data.time)-1)))
		self.plot2D = img

		self.plot2D.x_axis.title="Wavelength (nm)"
		self.plot2D.y_axis.title="Samples"

		if self.zoom_on==True:
			zoom = ZoomTool(component=img, tool_mode="box", always_on=True)
			img.overlays.append(zoom)

		img.tools.append(PanTool(img))

		self._plottimeindex_changed()
		self._plotwaveindex_changed()

	def _plottimeindex_changed(self):
		'''
		Plotting the 1D spectra chaco plot
		'''
		index_time_left=self.plottimeindex

		self.plottimeindexshow = Data.time[index_time_left]

		dw = ArrayPlotData(x=Data.wavelength,y=Data.TrA_Data[index_time_left,:])
		plot = Plot(dw)
		plot.plot(("x","y"), line_width=1)

		if self.zoom_on==True:
			zoom = ZoomTool(component=img, tool_mode="box", always_on=True)
			img.overlays.append(zoom)

		plot.tools.append(PanTool(plot))

		self.plotwavelength = plot
		self.plotwavelength.y_axis.title="Abs."
		self.plotwavelength.x_axis.title="Wavelength (nm)"

	def _plotwaveindex_changed(self):
		'''
		Plotting the 1D kinetic chaco plot
		'''

		index_wave_left=(abs(Data.wavelength-float(self.plotwaveindex))).argmin()

		self.plotwaveindexshow = Data.wavelength[index_wave_left]

		dt = ArrayPlotData(x=Data.time,y=Data.TrA_Data[:,index_wave_left])
		plot = Plot(dt)
		plot.plot(("x","y"), line_width=1)

		if self.zoom_on==True:
			zoom = ZoomTool(component=img, tool_mode="box", always_on=True)
			img.overlays.append(zoom)

		plot.tools.append(PanTool(plot))

		self.plottime = plot
		self.plottime.y_axis.title="Abs."
		self.plottime.x_axis.title="Time (ps)"

	def _Load_files_fired(self):
		'''
		Loads in all the files for PyTrA
		'''
		# Load TrA file into array depends on extension
		Data.filename = self.TrA_Raw_file
		TrA_Raw_file_name, TrA_Raw_file_extension = splitext(self.TrA_Raw_file)
		TrA_Raw_file_dir, TrA_Raw_file_name = split(self.TrA_Raw_file)
		TrA_Raw_name, TrA_Raw_ex = splitext(TrA_Raw_file_name)
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
			Data.TrA_Data=Data.TrA_Data[0:-2,:]
			Data.time=Data.time[0:-2]

		#Sort data into correct order
		inds = Data.wavelength.argsort()
		Data.TrA_Data = Data.TrA_Data[:,inds]
		Data.wavelength = Data.wavelength[inds]

		indst = Data.time.argsort()
		Data.TrA_Data = Data.TrA_Data[indst,:]
		Data.time = Data.time[indst]
		self.timelist = Data.time.tolist()

		self.wavelow = float(Data.wavelength[0])
		self.wavehigh = float(Data.wavelength[-1])
		self.waverange = len(Data.wavelength)
		self.timelow = int(0)
		self.timehigh = len(Data.time)-1
		self.timerange = len(Data.time)
		self.plottimeindex = float(Data.time[0])
		self.plotwaveindex = float(Data.wavelength[0])

		# Importing Chirp data

		try:
			Chirp_file_name, Chirp_file_extension = splitext(self.Chirp_file)

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
			self.log=("%s\nNo Chirp found"%(self.log))

		# Update chaco plot
		self._plot2Db_fired()

		self.log=("%s\nData file imported of size t=%s and wavelength=%s name=%s" %(self.log,Data.TrA_Data.shape[0],Data.TrA_Data.shape[1],TrA_Raw_name))

	def _showdata_fired(self):
		Dataviewer().edit_traits()

	def _open_csv_fired(self):
		'''
		Opens the file in the original
		'''
		startfile(self.TrA_Raw_file)

	def _Ohioloader_fired(self):
		ohio = OhioLoader().edit_traits()

		# Update chaco plot
		self._plot2Db_fired()

		self.log = ('%s\nData file imported of size %s by %s' %(self.log,Data.TrA_Data.shape[0],Data.TrA_Data.shape[1]))

	def _Shiftzero_fired(self):
		figure()
		contourf(Data.wavelength, Data.time[1:20], Data.TrA_Data[1:20,:], 100)
		title('Pick time zero')
		xlabel('Wavelength')
		ylabel('Time')
		fittingto = array(ginput(1))
		show()
		close()

		Data.time = Data.time-fittingto[0][1]

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\nShifted time by %s" %(self.log,fittingto[0][1])

	def _DeleteTraces_fired(self):
		figure()
		contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
		title('Pick between wavelength to delete (left to right)')
		xlabel('Wavelength')
		ylabel('Time')
		fittingto = array(ginput(2))
		show()
		close()

		index_wavelength_left=(abs(Data.wavelength-fittingto[0,0])).argmin()
		index_wavelength_right=(abs(Data.wavelength-fittingto[1,0])).argmin()+1

		Data.TrA_Data = delete(Data.TrA_Data,arange(index_wavelength_left,index_wavelength_right,1),1)
		Data.wavelength = delete(Data.wavelength,arange(index_wavelength_left,index_wavelength_right,1))

		self.wavelow = float(Data.wavelength[0])
		self.wavehigh = float(Data.wavelength[-1])
		self.waverange = len(Data.wavelength)

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\n \nDeleted traces %s and %s" %(self.log,fittingto[0,0],fittingto[1,0])

	def _Delete_spectra_fired(self):
		'''
		Delete rows
		'''
		figure()
		contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
		title('Pick between times to delete (top to bottom)')
		xlabel('Wavelength')
		ylabel('Time')
		fittingto = array(ginput(2))
		show()
		close()

		index_time_top=(abs(Data.time-fittingto[1,1])).argmin()
		index_time_bottom=(abs(Data.time-fittingto[0,1])).argmin()

		Data.TrA_Data = delete(Data.TrA_Data,arange(index_time_top,index_time_bottom,1),0)
		Data.time = delete(Data.time,arange(index_time_top,index_time_bottom,1))

		self.timelow = float(0)
		self.timehigh = len(Data.time)-1
		self.timerange = len(Data.time)

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\n \nDeleted spectra between %s and %s" %(self.log,fittingto[0,1],fittingto[1,1])

	def _DeleteSpectra_1_fired(self):
		'''
		Delete single spectrum
		'''

		index_time_left=self.plottimeindex
		time_val = Data.time[index_time_left]

		Data.TrA_Data = delete(Data.TrA_Data,index_time_left,0)
		Data.time = delete(Data.time,index_time_left)

		self.timelow = float(0)
		self.timehigh = len(Data.time)-1
		self.timerange = len(Data.time)

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\nDeleted spectrum %s" %(self.log,time_val)

	def _DeleteTraces_1_fired(self):
		'''
		Delete single trace
		'''

		index_wave_left=(abs(Data.wavelength-float(self.plotwaveindex))).argmin()

		wave_val = Data.wavelength[index_wave_left]

		Data.TrA_Data = delete(Data.TrA_Data,index_wave_left,1)
		Data.wavelength = delete(Data.wavelength,index_wave_left)

		self.wavelow = float(Data.wavelength[0])
		self.wavehigh = float(Data.wavelength[-1])
		self.waverange = len(Data.wavelength)

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\nDeleted trace at %s" %(self.log,wave_val)

	def _PlotChirp_fired(self):
		figure()
		contourf(Data.wavelength_C, Data.time_C, Data.Chirp, 100)
		title('Zoom into region you want to fit chirp')
		xlabel('Wavelength (nm)')
		ylabel('Time (ps)')
		show()

	def _Fix_Chirp_fired(self):

		try:
			ymin, ymax = ylim()

		except:
			self.log = "%s\nPlot the chirp and zoom into the area you want to fit to" %(self.log)

		#plot file and pick points for graphing
		figure(figsize=(20,12))
		title('Pick 8 points on the chirp')
		xlabel('Wavelength')
		ylabel('Time')
		contourf(Data.wavelength_C, Data.time_C, Data.Chirp, 20)
		ylim((ymin,ymax))
		polypts = array(ginput(8))
		show()
		close()

		#Fit a polynomial of the form p(x) = p[2] + p[1] + p[0]
		fitcoeff, residuals, rank, singular_values, rcond = polyfit(polypts[:, 0], polypts[:, 1], 2, full=True)

		stdev = sum(residuals**2)/8

		#finding where zero time is
		figure(figsize=(20,12))
		title("Pick point on wave front")
		xlabel('Wavelength')
		ylabel('Time')
		contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
		ylim((-2,2))
		fittingto = array(ginput(1)[0])
		show()
		close()

		#Moves the chirp inorder to correct coefficient
		fitcoeff[2] = (-fitcoeff[0]*fittingto[0]**2 - fitcoeff[1]*fittingto[0] + fittingto[1])

		#Iterate over the wavelengths and interpolate for the corrected values

		for i in range(0, len(Data.wavelength), 1):

			correcttimeval = polyval(fitcoeff, Data.wavelength[i])
			f = interpolate.interp1d((Data.time-correcttimeval), (Data.TrA_Data[:, i]), bounds_error=False, fill_value=0)
			fixed_wave = f(Data.time)
			Data.TrA_Data[:, i] = fixed_wave

		# Update chaco plot
		self._plot2Db_fired()

		self.log = "%s\n \nPolynomial fit with form %s*x^2 + %s*x + %s stdev %s" %(self.log,fitcoeff[0],fitcoeff[1],fitcoeff[2],stdev)

	def _Fit_Trace_fired(self):

		index_wavelength=(abs(Data.wavelength-self.plotwaveindex)).argmin()
		Data.tracefitmodel = fitgui.fit_data(Data.time,Data.TrA_Data[:,index_wavelength],autoupdate=False,model=Convoluted_exp1)

		results_error = Data.tracefitmodel.getCov().diagonal()
		results_par = Data.tracefitmodel.params
		results = Data.tracefitmodel.parvals

		self.log= ('%s\n \nFitted parameters at wavelength %s \nFitting parameters'%(self.log,Data.wavelength[index_wavelength]))

		# Update chaco plot
		self._plot2Db_fired()

		for i in range(len(results)):
			self.log = ('%s\n%s = %s +- %s'%(self.log,results_par[i],results[i],results_error[i]))

	def _Fit_Spec_fired(self):

		index_time=self.plottimeindex
		Data.tracefitmodel = fitgui.fit_data(Data.wavelength,Data.TrA_Data[index_time,:],autoupdate=False,model=Gauss_1)

		results_error = Data.tracefitmodel.getCov().diagonal()
		results_par = Data.tracefitmodel.params
		results = Data.tracefitmodel.parvals

		self.log= ('%s\n \nFitted parameters at time %s \nFitting parameters'%(self.log,Data.time[index_time]))

		for i in range(len(results)):
			self.log = ('%s\n%s = %s +- %s'%(self.log,results_par[i],results[i],results_error[i]))

	def _mcmc_fired(self):
		mcmc_app = mcmc.MCMC_1(parameters=[ mcmc.Params(name=i) for i in Data.tracefitmodel.params])
		mcmc_app.edit_traits(kind='livemodal')
		mcmc_app = mcmc.MCMC_1(parameters=[])
		self.log = ('%s\n \n---MCMC sampler summary (pymc)---\nBayesian Information Criterion = %s'%(self.log,Data.mcmc['MAP']))
		for i in Data.tracefitmodel.params:
			self.log = ('%s\n%s,mean %s,stdev %s'%(self.log,i,Data.mcmc['MCMC'].stats()[i]['mean'],Data.mcmc['MCMC'].stats()[i]['standard deviation']))
		self.log = ('%s\nsigma,mean %s,stdev %s'%(self.log,Data.mcmc['MCMC'].stats()['sig']['mean'],Data.mcmc['MCMC'].stats()['sig']['standard deviation']))

	def _SVD_fired(self):

		xmin=self.plot2D.range2d.x_range.low
		xmax=self.plot2D.range2d.x_range.high
		ymin=self.plot2D.range2d.y_range.low
		ymax=self.plot2D.range2d.y_range.high

		index_wavelength_left=(abs(Data.wavelength-xmin)).argmin()
		index_wavelength_right=(abs(Data.wavelength-xmax)).argmin()

		index_time_left=(abs(Data.time-ymin)).argmin()
		index_time_right=(abs(Data.time-ymax)).argmin()

		U, s, V_T = linalg.svd(Data.TrA_Data[index_time_left:index_time_right,index_wavelength_left:index_wavelength_right])

		f=figure()
		f.text(0.5,0.975,("SVD %s" %(self.title)),horizontalalignment='center',verticalalignment='top')
		subplot(341)
		plot(Data.time[index_time_left:index_time_right],U[:,0])
		title("1")
		xlabel("time (ps)")
		ylabel("abs.")
		subplot(342)
		plot(Data.time[index_time_left:index_time_right],U[:,1])
		title("2")
		xlabel("time (ps)")
		ylabel("abs.")
		subplot(343)
		plot(Data.time[index_time_left:index_time_right],U[:,2])
		title("3")
		xlabel("time (ps)")
		ylabel("abs.")
		subplot(344)
		plot(Data.time[index_time_left:index_time_right],U[:,3])
		title("4")
		xlabel("time (ps)")
		ylabel("abs.")
		subplot(345)
		plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[0,:])
		title("%s" %(s[0]))
		xlabel("wavelength (nm)")
		ylabel("abs.")
		subplot(346)
		plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[1,:])
		title("%s" %(s[1]))
		xlabel("wavelength (nm)")
		ylabel("abs.")
		subplot(347)
		plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[2,:])
		title("%s" %(s[2]))
		xlabel("wavelength (nm)")
		ylabel("abs.")
		subplot(348)
		plot(Data.wavelength[index_wavelength_left:index_wavelength_right],V_T[3,:])
		title("%s" %(s[3]))
		xlabel("wavelength (nm)")
		ylabel("abs.")
		subplot(349)
		[SVD_1_x,SVD_1_y]=meshgrid(V_T[0,:],U[:,0])
		SVD_1 = multiply(SVD_1_x,SVD_1_y)*s[0]
		contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_1,50)
		subplot(3,4,10)
		[SVD_2_x,SVD_2_y]=meshgrid(V_T[1,:],U[:,1])
		SVD_2 = multiply(SVD_2_x,SVD_2_y)*s[1]
		contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_2,50)
		subplot(3,4,11)
		[SVD_3_x,SVD_3_y]=meshgrid(V_T[2,:],U[:,2])
		SVD_3 = multiply(SVD_3_x,SVD_3_y)*s[2]
		contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_3,50)
		subplot(3,4,12)
		[SVD_4_x,SVD_4_y]=meshgrid(V_T[3,:],U[:,3])
		SVD_4 = multiply(SVD_4_x,SVD_4_y)*s[3]
		contourf(Data.wavelength[index_wavelength_left:index_wavelength_right],Data.time[index_time_left:index_time_right],SVD_4,50)
		subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.94, wspace=0.2, hspace=0.2)
		show()

		figure()
		semilogy(s[0:9],'*')
		title("First 10 singular values")
		show()

		self.log = "%s\nFirst 5 singular values %s in range wavelength %s to %s, time %s to %s" %(self.log,s[0:5], xmin, xmax, ymin, ymax)

	def _EFA_fired(self):

		#number of singular values to track
		singvals = 3

		#Time
		rows = Data.TrA_Data.shape[0]
		forward_r = zeros((rows,singvals))
		backward_r = zeros((rows,singvals))

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

		figure()
		semilogy(Data.time[singvals:],forward_r[singvals:,:],'b',Data.time[:(rows-singvals)],backward_r[:(rows-singvals),:],'r')
		title("%s EFA time" %(self.title))
		xlabel("Time (ps)")
		ylabel("Log(EV)")
		show()

		#Wavelength

		cols = Data.TrA_Data.shape[1]
		forward_c = zeros((cols,singvals))
		backward_c = zeros((cols,singvals))

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

		figure()
		semilogy(Data.wavelength[singvals:],forward_c[singvals:,:],'b',Data.wavelength[:cols-singvals],backward_c[:cols-singvals,:],'r')
		title("%s EFA wavelength" %(self.title))
		xlabel("Wavelength (nm)")
		ylabel("Log(EV)")
		show()

	def _Multiple_Trace_fired(self):
		self.Traces_num = 0
		Data.Traces = 0

		figure(figsize=(15,10))
		contourf(Data.wavelength, Data.time, Data.TrA_Data, 100)
		title('Pick between wavelength to fit (left to right)')
		xlabel('Wavelength')
		ylabel('Time')
		fittingto = array(ginput(2))
		show()
		close()

		index_wavelength_left=(abs(Data.wavelength-fittingto[0,0])).argmin()
		index_wavelength_right=(abs(Data.wavelength-fittingto[1,0])).argmin()

		Data.Traces = Data.TrA_Data[:,index_wavelength_left:index_wavelength_right].transpose()

		self.log= '%s\n\n%s Traces saved from %s to %s' %(self.log,Data.Traces.shape[0], fittingto[0,0], fittingto[1,0])

	def grid_data(self):
		'''
		Populate the Data.TrA_Data_gridded for 3D plot and chaco plot
		'''
		Data.TrA_Data_gridded = array([])

		#Gets smallest spacing to use to construct mesh
		y_step = Data.time[(abs(Data.time-0)).argmin()+1]-Data.time[(abs(Data.time-0)).argmin()]

		x = linspace(Data.wavelength[0],Data.wavelength[-1],len(Data.wavelength))
		y = arange(Data.time[0], Data.time[-1],y_step)

		[xi,yi] = meshgrid(x,y)

		Data.xi = xi
		Data.yi = yi

		Data.TrA_Data_gridded = xi

		vectors = array([[0],[0],[0]])

		# Create the x, y, z vectors
		for i in range(len(Data.wavelength)):
			repeating_wavelength = array(ones((len(Data.time)))*Data.wavelength[i])
			vectors_temp = array([Data.time,repeating_wavelength,Data.TrA_Data[:,i]])
			if i==0:
				vectors = vectors_temp
			else:
				vectors = hstack((vectors, vectors_temp))

		Data.TrA_Data_gridded = interpolate.griddata((vectors[1,:],vectors[0,:]),vectors[2,:],(xi,yi), method='linear', fill_value=0)

	def _Plot_3D_fired(self):

		self.scene.mlab.clf()
		self.grid_data()

		#Sends 3D plot to mayavi in gui

		#uncomment for plotting actual data matrix
		#self.scene.mlab.surf(Data.time,Data.wavelength,Data.TrA_Data,warp_scale=-self.z_height*100)
		#gridded plot which gives correct view
		self.scene.mlab.surf(Data.yi,Data.xi,Data.TrA_Data_gridded, warp_scale=-self.z_height*100)
		self.scene.mlab.colorbar(orientation="vertical")
		self.scene.mlab.axes(nb_labels=5,)
		self.scene.mlab.ylabel("wavelength (nm)")
		self.scene.mlab.xlabel("time (ps)")

	def _z_height_changed(self):
		# Only redraws it so does not have to regrid
		self.scene.mlab.clf()
		self.scene.mlab.surf(Data.yi,Data.xi,Data.TrA_Data_gridded, warp_scale=-self.z_height*100)
		self.scene.mlab.colorbar(orientation="vertical")
		self.scene.mlab.axes(nb_labels=5,)
		self.scene.mlab.axes(z_axis_visibility=False)
		self.scene.mlab.ylabel("wavelength (nm)")
		self.scene.mlab.xlabel("time (ps)")

	def _Plot_2D_fired(self):
		figure()
		contourf(Data.wavelength, Data.time, Data.TrA_Data, 200)
		xlabel('Wavelength (nm)')
		ylabel('Times (ps)')
		title(self.title)
		colorbar()
		show()

	def _Plot_Traces_fired(self):
		figure(figsize=(15,10))
		plot(Data.time, Data.Traces.transpose() )
		title("%s Traces" %(self.title))
		xlabel('Time')
		ylabel('Abs')
		show()

	def _Kinetic_Trace_fired(self):

		index_wavelength=(abs(Data.wavelength-float(self.plotwaveindex))).argmin()

		figure(figsize=(20,12))
		plot(Data.time, Data.TrA_Data[:,index_wavelength] )
		title("%s %s" %(self.title, Data.wavelength[index_wavelength]))
		xlabel('Time')
		ylabel('Abs')
		show()

	def _Spectra_fired(self):

		index_time=self.plottimeindex

		figure()
		plot(Data.wavelength, Data.TrA_Data[index_time,:] )
		title("%s %s" %(self.title, Data.time[index_time]))
		xlabel('Wavelength')
		ylabel('Abs')
		show()

	def _multiple_plots_fired(self):

		xmin=self.plot2D.range2d.x_range.low
		xmax=self.plot2D.range2d.x_range.high
		ymin=self.plot2D.range2d.y_range.low
		ymax=self.plot2D.range2d.y_range.high

		index_wavelength_left=(abs(Data.wavelength-xmin)).argmin()
		index_wavelength_right=(abs(Data.wavelength-xmax)).argmin()

		index_time_left=(abs(Data.time-ymin)).argmin()
		index_time_right=(abs(Data.time-ymax)).argmin()

		indexwave = int((index_wavelength_right-index_wavelength_left)/10)

		# spectrum from every 10th spectra

		timevec = ones([Data.time[index_time_left:index_time_right].shape[0],10])
		time = ones([Data.time[index_time_left:index_time_right].shape[0],10])
		wavelengthvals = ones(10)

		for i in range(10):
			timevec[:,i] = average(Data.TrA_Data[index_time_left:index_time_right,index_wavelength_left+((i)*indexwave):index_wavelength_left+((i)*indexwave)+indexwave],axis=1)
			time[:,i] = Data.time[index_time_left:index_time_right]
			wavelengthvals[i] = round(average(Data.wavelength[index_wavelength_left+((i)*indexwave):index_wavelength_left+((i)*indexwave)+indexwave]),1)

		figure()
		colormap = cm.jet
		gca().set_color_cycle([colormap(i) for i in linspace(0, 0.9, 10)])
		plot(time,timevec )
		legend(wavelengthvals)
		xlabel('Time (ps)')
		ylabel('Abs.')
		title("Averaged %s %s" %(self.title, 'Wavelengths (nm)'))
		show()

		indextime = int((index_time_right-index_time_left)/10)

		wavevec = ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
		wave = ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
		timevals = ones(10)

		for i in range(10):
			wavevec[:,i] = average(Data.TrA_Data[index_time_left+((i)*indextime):index_time_left+((i)*indextime)+indextime,index_wavelength_left:index_wavelength_right],axis=0)
			wave[:,i] = Data.wavelength[index_wavelength_left:index_wavelength_right]
			timevals[i] = round(average(Data.time[index_time_left+((i)*indextime):index_time_left+((i)*indextime)+indextime]),1)

		figure()
		colormap = cm.jet
		gca().set_color_cycle([colormap(i) for i in linspace(0, 0.9, 10)])
		plot(wave,wavevec )
		legend(timevals)
		title("Averaged %s %s" %(self.title, 'Times (ps)'))
		xlabel('Wavelength (nm)')
		ylabel('Abs.')
		show()

	def _Normalise_fired(self):

		xmin=self.plot2D.range2d.x_range.low
		xmax=self.plot2D.range2d.x_range.high
		ymin=self.plot2D.range2d.y_range.low
		ymax=self.plot2D.range2d.y_range.high

		index_wavelength_left=(abs(Data.wavelength-xmin)).argmin()
		index_wavelength_right=(abs(Data.wavelength-xmax)).argmin()

		index_time_left=(abs(Data.time-ymin)).argmin()
		index_time_right=(abs(Data.time-ymax)).argmin()

		indextime = int((index_time_right-index_time_left)/10)

		wavevec = ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
		wave = ones([Data.wavelength[index_wavelength_left:index_wavelength_right].shape[0],10])
		timevals = ones(10)

		for i in range(10):
			wavevec[:,i] = Data.TrA_Data[(index_time_left+((i)*indextime)),index_wavelength_left:index_wavelength_right]
			max_i = max(wavevec[:,i])
			min_i = min(wavevec[:,i])
			wavevec[:,i] = (wavevec[:,i]-min_i)/(max_i-min_i)
			wave[:,i] = Data.wavelength[index_wavelength_left:index_wavelength_right]
			timevals[i] = Data.time[index_time_left+((i)*indextime)]

		figure()
		colormap = cm.jet
		gca().set_color_cycle([colormap(i) for i in linspace(0, 0.9, 10)])
		plot(wave,wavevec)
		legend(timevals)
		title("Normalised %s %s" %(self.title, 'Times (ps)'))
		xlabel('Wavelength (nm)')
		ylabel('Abs.')
		show()

		indexwave = int((index_wavelength_right-index_wavelength_left)/10)

		# spectrum from every 10th spectra

		timevec = ones([Data.time[index_time_left:index_time_right].shape[0],10])
		time = ones([Data.time[index_time_left:index_time_right].shape[0],10])
		wavelengthvals = ones(10)

		for i in range(10):
			timevec[:,i] = Data.TrA_Data[index_time_left:index_time_right,(index_wavelength_left+((i)*indexwave))]
			max2_i = max(timevec[:,i])
			min2_i = min(timevec[:,i])
			timevec[:,i] = (timevec[:,i]-min2_i)/(max2_i-min2_i)
			time[:,i] = Data.time[index_time_left:index_time_right]
			wavelengthvals[i] = Data.wavelength[index_wavelength_left+((i)*indexwave)]

		figure()
		colormap = cm.jet
		gca().set_color_cycle([colormap(i) for i in linspace(0, 0.9, 10)])
		plot(time,timevec )
		legend(wavelengthvals)
		xlabel('Time (ps)')
		ylabel('Abs.')
		title("Normalised %s %s" %(self.title, 'Wavelengths (nm)'))
		show()

	def _savetraces_fired(self):
		try:
			f=open(("%s\Traces.txt" %(dirname(self.TrA_Raw_file))), 'w')
			for i in range(len(Data.time)):
				f.write("%s" %(Data.time[i]))
				for j in range(len(Data.Traces)):
					f.write(",%s" %(Data.Traces[j,i]))
				f.write("\n")
			f.close()
			self.log = '%s\n\nTraces saved to %s' %(self.log,dirname(self.TrA_Raw_file))
		except:
			self.log = '%s\n\nPlease select multiple traces' %(self.log)

	def _Save_Glo_fired(self):
		# Generates ouput file in Glotaran Time explicit format
		pathname = "%s\Glotaran.txt" %(dirname(self.TrA_Raw_file))
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

		self.log = '%s \nSaved Glotaran file to TrA data file directory' %(self.log)

	def _Trace_Igor_fired(self):

		import win32com.client # Communicates with Igor needs pywin32 library
		f=open(("%s\Traces.txt" %(dirname(self.TrA_Raw_file))), 'w')
		for i in range(len(Data.time)):
			f.write("%s" %(Data.time[i]))
			for j in range(len(Data.Traces)):
				f.write(",%s" %(Data.Traces[j,i]))
			f.write("\n")
		f.close()
		# Sends traces to Igor and opens up Global fitting gui in Igor
		igor=win32com.client.Dispatch("IgorPro.Application")

		#Load into igor using LoadWave(/A=Traces/J/P=pathname) /J specifies it as a txt delimited file
		igor.Execute('NewPath pathName, "%s"' %(dirname(self.TrA_Raw_file)))
		igor.Execute('Loadwave/J/P=pathName "Traces.txt"')
		igor.Execute('Rename wave0,timeval')

		# Run global fitting gui in Igor
		igor.Execute('WM_NewGlobalFit1#InitNewGlobalFitPanel()')
		igor.clear()

	def _Save_csv_fired(self):
		now = date.today()
		pathname = "%s\Saved%s%s.csv" %(dirname(self.TrA_Raw_file), now.strftime("%m-%d-%y"),self.title)
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

		self.log= '%s\n\nSaved to TrA data file directory' %(self.log)

	def _Save_log_fired(self):
		now = date.today()
		pathname = "%s\log%s_%s.log" %(dirname(self.TrA_Raw_file), now.strftime("%m-%d-%y"),self.title)
		f = open(pathname, 'w')
		f.write("%s"%(self.log))

		self.log= '%s\n\nSaved log file to %s' %(self.log,dirname(self.TrA_Raw_file))

	def _Help_fired(self):
		help = Help().edit_traits()

main = MainWindow()

if __name__=='__main__':
	main.configure_traits()
