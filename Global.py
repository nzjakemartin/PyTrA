from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt
import pymodelfit as fit
from scipy.optimize import fmin
import numpy as np


def Global_chi2(start,linked,fix_vals,fixed):
	chi2 = 0.0
	for i in range(len(start)-1):
		Data.tracefitmodel.pardict[linked[i]] = start[i]
	for i in range(len(fix_vals)-1):
		Data.tracefitmodel.pardict[fixed[i]] = fix_vals[i]
	fix_ = linked
	start = np.array(start)
	print Data.Traces.shape
	for i in range(len(Data.Traces[:,1])):
		Data.tracefitmodel.fitData(Data.time,Data.Traces[i,:],fixedpars=fix_)
		chi2 = (Data.tracefitmodel.chi2Data()[0]+chi2)/2
	return(chi2)


class Global(HasTraits):
	parameters = List
	max_iter = Int(400)
	fit = Button("Fit")
	status = Str
	linked = []
	fixed = []
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

		#Use minimisation over linked parameters
		if self.sample == 'fmin':
			xopt = fmin(Global_chi2, start, args=(linked,fix_vals,fixed))

		self.status = xopt

		#Use MCMC over linked parameters
		if self.sample == 'mcmc':
			loop.MCMC

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