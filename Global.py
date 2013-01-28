from traits.api import  HasTraits, File, Button, Array, Enum, Instance, Str, List, HasPrivateTraits, Float, Int, Bool
from traitsui.api import Group, Item, View, Label, HSplit, Tabbed, ListEditor
from Data import Data
import matplotlib.pyplot as plt
import pymodelfit as fit
from scipy.optimize import fmin
import numpy as np

class Fitting():
    linked = []
    fixed = []
    start = np.array([])
    fix_vals = np.array([])

    def Global(self):
        stdev = 0.0
        print self.linked, self.fixed, self.start, self.fix_vals
        for i in range(len(self.start)-1):
            Data.tracefitmodel.pardict[self.linked[i]] = self.start[i]
        for i in range(len(self.fix_vals)-1):
            Data.tracefitmodel.pardict[self.fixed[i]] = self.fix_vals[i]
        fix_ = self.linked
        start = np.array(self.start)
        print start
        for i in range(len(Data.Traces[:,1])):
            Data.tracefitmodel.fitData(Data.time,Data.Traces[i,:],fixedpars=fix_)
            stdev = (Data.tracefitmodel.stdData()+stdev)/2
        return(stdev)

    def MCMC(self):
        self.fixed = self.linked.append(self.fixed)
        print 'Too be completed'


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

        loop = Fitting()
        loop.linked=self.linked
        loop.fixed=self.fixed
        loop.start=start
        loop.fix_vals=fix_vals

        print loop.start

        #Use minimisation over linked parameters
        if self.sample == 'fmin':
            xopt, fopt = fmin(loop.Global, loop.start, full_output=1)
            self.status = xopt,fopt

        print xopt, fopt

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