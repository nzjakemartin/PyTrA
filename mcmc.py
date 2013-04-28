from pymc import *
from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt

class MCMC_1(HasTraits):
	parameters = List
	iter = Int(10000)
	thin = Int(2)
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
		Item('thin'),
		Item('run'),
		Label('See log file for ouput'),
		title   = 'MCMC', resizable=True,
		buttons = [ 'OK', 'Cancel' ]
	)

	def _run_fired(self):
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

		x,y,w = Data.tracefitmodel.data
		model = Data.tracefitmodel.getMCMC(x,y,priors=self.prior_dict,datamodel=None)
		model_fit = MAP(model)
		Data.mcmc = {}
		model_fit.fit()
		Data.mcmc['MAP'] = model_fit.BIC
		MC = pymc.MCMC(model_fit.variables)
		MC.sample(self.iter,burn=self.burn_in,thin=self.thin)
		#for i in Data.tracefitmodel.parms:
		#	Data.mcmc[i] = MC.stats()[i]
		Data.mcmc['MCMC'] = MC

		for i in range(len(self.parameters)):
			if self.parameters[i].plot==True:
				Matplot.plot(MC.trace(self.parameters[i].name))

		plt.show()

class Params(HasPrivateTraits):
	#Name of string
	i = Int
	name = Str
	min = Float
	max = Float
	sig = Float
	plot = Bool(False)
	dist = Enum('Uniform','Normal','Poisson')

	view = View(
		Item('name',label='Parameter',style='readonly'),
		Item('min',label='Lower value for Uniform'),
		Item('max',label='Upper value for Uniform'),
		Item('sig',label='Sigma value for Normal'),
		Item('dist',label='Distribution for parameter'),
		Item('plot',label='plot')
	)