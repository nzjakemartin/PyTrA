from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt
import pymodelfit as fit
from scipy.optimize import fmin
import numpy as np


def Global_chi2(start,linked,fix_vals,fixed):

	#Short term solution to changing parameter values

	if any('Tone' in s for s in linked):
		x = linked.index('Tone')
		Data.tracefitmodel.Tone = start[x]

	if any('Ttwo' in s for s in linked):
		x = linked.index('Two')
		Data.tracefitmodel.Ttwo = start[x]

	if any('Tthree' in s for s in linked):
		x = linked.index('Tthree')
		Data.tracefitmodel.Tthree = start[x]

	if any('Tfour' in s for s in linked):
		x = linked.index('Tfour')
		Data.tracefitmodel.Tfour = start[x]

	if any('Aone' in s for s in linked):
		x = linked.index('Aone')
		Data.tracefitmodel.Aone = start[x]

	if any('Atwo' in s for s in linked):
		x = linked.index('Atwo')
		Data.tracefitmodel.Atwo = start[x]

	if any('Athree' in s for s in linked):
		x = linked.index('Athree')
		Data.tracefitmodel.Athree = start[x]

	if any('Afour' in s for s in linked):
		x = linked.index('Afour')
		Data.tracefitmodel.Afour = start[x]

	if any('fwhm' in s for s in linked):
		x = linked.index('fwhm')
		Data.tracefitmodel.fwhm = start[x]

	if any('c' in s for s in linked):
		x = linked.index('c')
		Data.tracefitmodel.c = start[x]

	if any('mu' in s for s in linked):
		x = linked.index('mu')
		Data.tracefitmodel.mu = start[x]

	#Need to fix
	#for i in range(len(fix_vals)):
	#	Data.tracefitmodel.pardict[fixed[i]] = fix_vals[i]

	fix_ = linked

	start = np.array(start)
	chi2 = np.ones(len(Data.Traces[:,1]))

	for i in range(len(Data.Traces[:,1])):
		Data.tracefitmodel.fitData(Data.time,Data.Traces[i,:],fixedpars=fix_,)
		plt.figure()
		plt.title('%s'%(Data.tracefitmodel.chi2Data()[0]))
		Data.tracefitmodel.plot()
		plt.show()
		chi2[i] = Data.tracefitmodel.chi2Data()[0]
		if chi2[i]**2 >= 1:
			chi2[i]=0

	return np.average(chi2)


class Global(HasTraits):
	parameters = List
	max_iter = Int(400)
	fit = Button("Fit")
	status = Str
	linked = []
	fixed = []
	range_start = Array(np.float,(1,2))
	grid_size = Float(0.01)
	sample = Enum('fmin', 'mcmc')

	view = View(
		Item( 'parameters',
			style  = 'custom',
			editor = ListEditor( use_notebook = True,
				deletable    = True,
				dock_style   = 'tab',
				page_name    = '.name' )
		),

		Item('max_iter'),
		Item('fit'),
		Item('sample'),
		Item('status'),
		Item('range_start'),
		Item('grid_size'),
		title   = 'Global', resizable=True,
		buttons = [ 'OK', 'Cancel' ]
	)

	def _fit_fired(self):

		start = []
		fix_vals = []

		#Construct list of fixed and linked parameters

		for i in range(len(self.parameters)):
			if self.parameters[i].linked==True:
				self.linked.append(self.parameters[i].name)
				start.append(self.parameters[i].val)
			if self.parameters[i].fixed==True:
				self.fixed.append(self.parameters[i].name)
				fix_vals.append(self.parameters[i].val)

		linked=self.linked
		fixed=self.fixed

		#Visual error surface as it is scanned through

		start_scan = np.arange(self.range_start[0][0],self.range_start[0][1],self.grid_size)
		chi2_scan = np.ones(len(start_scan))
		print chi2_scan.shape, start_scan.shape

		i = 0

		for start_scan_loop in start_scan:
			chi2_scan[i] = Global_chi2([start_scan_loop], linked, fix_vals,fixed)
			print chi2_scan[i]
			i+=1

		plt.figure()
		plt.plot(start_scan,chi2_scan)
		plt.show()

		#Use minimisation over linked parameters
		#if self.sample == 'fmin':
		#	xopt = fmin(Global_chi2, start, args=(linked,fix_vals,fixed))

		#self.status = ('minimised value :%s'%(xopt))

		#Use MCMC over linked parameters
		if self.sample == 'mcmc':
			loop.MCMC

		print self.linked, self.fixed

		self.linked = []
		self.fixed = []
		start = []
		fix_vals = []

class Params(HasPrivateTraits):
	#Name of string
	i = Int
	name = Str
	fixed = Bool
	linked = Bool
	val = Float

	view = View(
		Item('name',label='Parameter',style='readonly'),
		Item('fixed',label='fix'),
		Item('linked',label='link'),
		Item('val',label='Value'),
	)

	def _val_default(self):
		return Data.tracefitmodel.pardict[self.name]