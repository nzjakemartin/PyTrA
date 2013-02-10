from pymc import *
from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt

class MCMC(HasTraits):
	parameters = List
	iter = Int(10000)
	burn_in = Int(0)
	run = Button("Run")
	prior_dict = dict()
	status = Str('')

	view = View(
		Item( 'parameters',
			style  = 'custom',
			editor = ListEditor( use_notebook = True,
				deletable    = True,
				dock_style   = 'tab',
				page_name    = '.name' )
		),

		Item('iter'),
		Item('burn_in'),
		Item('run'),
		Label('See log file for ouput'),
		title   = 'MCMC', resizable=True,
		buttons = [ 'OK', 'Cancel' ]
	)

	def _run_fired(self):

		"""

		"""
		for i in range(len(self.parameters)):

			#Sets a Poisson Distribution
			if self.parameters[i].dist=='Poisson':
				self.prior_dict[self.parameters[i].name]=0

			#Sets a Normal Distribution
			if self.parameters[i].dist=='Uniform':
				self.prior_dict[self.parameters[i].name]=(self.parameters[i].min,self.parameters[i].max)

			#Sets a Normal Distribution
			if self.parameters[i].dist=='Normal':
				self.prior_dict[self.parameters[i].name]=self.parameters[i].sig

		print "dictionary created"
		print self.prior_dict

		x,y,w = Data.tracefitmodel.data
		model = Data.tracefitmodel.getMCMC(x,y,priors=self.prior_dict,datamodel=None)

		model.sample(self.iter,burn=self.burn_in)

		Matplot.plot(model)
		plt.show()
		model.stats()

		return(self.status)

class Params(HasPrivateTraits):
	#Name of string
	i = Int
	name = Str
	min = Float
	max = Float
	sig = Float
	dist = Enum('Uniform','Normal','Poisson')

	view = View(
		Item('name',label='Parameter',style='readonly'),
		Item('min',label='Lower value for Uniform'),
		Item('max',label='Upper value for Uniform'),
		Item('sig',label='Sigma value for Normal'),
		Item('dist',label='Distribution for parameter'),
	)