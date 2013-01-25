from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt
import pymodelfit as fit
from scipy.optimize import fmin

class Global(HasTraits):
    parameters = List
    max_iter = Int(400)
    fit = Button("Fit")
    status = Str
    linked = []
    fixed = []
    sample = Enum('fmin','mcmc')

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
        title   = 'Global', resizable=True,
        buttons = [ 'OK', 'Cancel' ]
    )

    def _fit_fired(self):

        def fitting(fix):
            stdev = 0.0
            for i in range(len(Data.Traces[:,1])):
                Data.tracefitmodel.fitData(Data.time,Data.Traces,fixedpars=fix)
                stdev = (Data.tracefitmodel.stdData()+stdev)/2
            return(stdev)

        def Global(linked):
            fix = linked.append(fixed)
            return fitting(fix)

        #Construct list of fixed and linked parameters

        for i in range(len(self.parameters)):
            if self.parameters[i].linked==True:
                self.linked.append(self.parameters[i].name)
            if self.parameters[i].fixed==True:
                self.fixed.append(self.parameters[i].name)

        #Use minimisation over linked parameters

        xopt = fmin(Global, linked)



        #Use MCMC over linked parameters

        #Compute error of the result

        #Print out results in log file


        print "dictionary created"
        print self.prior_dict

        x,y,w = Data.tracefitmodel.data
        model = Data.tracefitmodel.getMCMC(x,y,priors=self.prior_dict,datamodel=None)

        model.sample(self.iter,burn=self.burn_in)

class Params(HasPrivateTraits):
    #Name of string
    i = Int
    name = Str
    fixed = Bool
    linked = Bool
    dist = Enum('Normal','Uniform','Poisson')

    view = View(
        Item('name',label='Parameter',style='readonly'),
        Item('fixed',label='fix parameter'),
        Item('linked',label='link parameter'),
        Item('dist',label='Distribution for parameter'),
    )