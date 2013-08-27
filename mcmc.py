from pymc import MCMC, Matplot, geweke, raftery_lewis, MAP, distributions
import pymc
from traits.api import  HasTraits, Button, Enum, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Item, View, Label, HSplit, ListEditor
from Data import Data
from matplotlib.pyplot import show
from inspect import getargspec
from types import MethodType

class MCMC_1(HasTraits):
	parameters = List
	iter = Int(10000)
	thin = Int(2)
	burn_in = Int(0)
	sigmin = Float(0.0)
	sigmax = Float(1.0)
	run = Button("Run")
	plot = Bool(True)
	plotg = Bool(False)
	status = Str('')
	Information = Str('Output')
	prior_dict = {}

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
		Label('Sigma min and max value'),
		HSplit(Item('sigmin'),Item('sigmax')),
		Item('plot',label='Distribution plot'),
		Item('plotg',label='Geweke plot'),
		Item('Information',style='custom'),
		title   = 'MCMC',
		resizable=True,
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

		sig_val = (self.sigmin,self.sigmax)
		sig = distributions.Uniform('sig',sig_val[0],sig_val[1])

		x,y,w = Data.tracefitmodel.data
		model = Data.tracefitmodel.getMCMC(x,y,priors=self.prior_dict,datamodel=(distributions.Normal,'mu',dict(tau=1/sig/sig)))
		model_fit = MAP(model)

		Data.mcmc = {}
		model_fit.fit()
		MC = MCMC(model_fit.variables)
		MC.sample(self.iter,burn=self.burn_in,thin=self.thin)
		Data.mcmc['MCMC'] = MC

		Data.mcmc['MAP'] = model_fit.BIC

		#Plotting
		if self.plot == True:
			Matplot.plot(MC,last=False)

		if self.plotg ==True:
			scores = geweke(MC, intervals=20)
			Matplot.geweke_plot(scores)
		show()

		Data.mcmc['raftery_lewis'] = raftery_lewis(MC, q=0.025, r=0.01, verbose=0)

		#Calculating maximum of the Raftery Lewis diagnostics

		thin = 0
		burn_in = 0
		iterations = 0

		#Finding maximum values for Raftery Lewis statistics
		for i,v in Data.tracefitmodel.pardict.iteritems():
			if iterations == 0:
				thin = Data.mcmc['raftery_lewis'][i][4]
				burn_in = Data.mcmc['raftery_lewis'][i][2]
				iterations = Data.mcmc['raftery_lewis'][i][3]
			else:
				if Data.mcmc['raftery_lewis'][i][4]>thin:
					thin = Data.mcmc['raftery_lewis'][i][4]
				if Data.mcmc['raftery_lewis'][i][2]>burn_in:
					burn_in = Data.mcmc['raftery_lewis'][i][2]
				if Data.mcmc['raftery_lewis'][i][3]>iterations:
					iterations = Data.mcmc['raftery_lewis'][i][3]
		#Sigma
		if Data.mcmc['raftery_lewis']['sig'][4]>thin:
			thin = Data.mcmc['raftery_lewis']['sig'][4]
		if Data.mcmc['raftery_lewis']['sig'][2]>burn_in:
			burn_in = Data.mcmc['raftery_lewis']['sig'][2]
		if Data.mcmc['raftery_lewis']['sig'][3]>iterations:
			iterations = Data.mcmc['raftery_lewis']['sig'][3]

		self.Information = 'Bayesian Information Criterion = %s' %(Data.mcmc['MAP'])
		self.Information ='%s\n---Raftery Lewis Diagnostics--- \n\nfor probability s=0.95, desired accuracy r=0.01 need;' % self.Information
		self.Information = '%s\nThinning = %s' %(self.Information,thin)
		self.Information = '%s\nBurn in = %s' %(self.Information,burn_in)
		self.Information = '%s\nIterations = %s' %(self.Information,iterations)


class Params(HasPrivateTraits):
	#Name of string
	i = Int
	name = Str
	min = Float
	max = Float
	sig = Float
	dist = Enum('Uniform','Normal')

	view = View(
		Item('name',label='Parameter',style='readonly'),
		Item('min',label='Lower value for Uniform'),
		Item('max',label='Upper value for Uniform'),
		Item('sig',label='Sigma value for Normal'),
		Item('dist',label='Distribution for parameter'),
		resizable=False
	)