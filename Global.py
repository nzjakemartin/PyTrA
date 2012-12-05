class Globalfit(HasTraits):
   Kinetic=Array(np.float,(1,3))
   IRF_para=Array(np.float,(2,1))
   Opt_steps=Array(np.float,(1,1))
   Run=Button("Run global fit")
   Results = String()
   
   view =View(
           Item('Kinetic', label='Kinetics guess'),
           Item('IRF_para', label='IRF guess (position width)'),
           Item('Opt_steps', label='Optimisation steps'),
           Item('Run'),
           Item('Results', show_label=True,springy=True,style='custom'),
           title   = 'PFfit', resizable=True,
           buttons = [ 'OK', 'Cancel' ]
           )
    
   def _Run_fired(self):
       
       os.startfile("C:\\Program Files\\R\\R-2.15.0\\bin\\i386\\Rserve.exe")
       
       Rconn = pyRserve.connect(host='localhost', port=6311)
       
       # Generates files for 
       pathname = "timp_file.txt"
       f = open(pathname, 'w')
       f.write("\n")
       f.write("\n")
       f.write("Time explicit\n")
       f.write("Intervalnr %d\n" %(len(Data.time)))
       f.write("\t")
       for i in range(len(Data.time)):
           f.write(" %s" %(Data.time[i]))
       f.write("\n")
       for i in range(len(Data.wavelength)):
           f.write(" %s" %(Data.wavelength[i]))
           for j in range(len(Data.time)):
               f.write(" %s" %(Data.TrA_Data[j,i]))
           f.write("\n")
       
       print "test"
       print self.Kinetic
       # import TIMP library
       Rconn("require('TIMP')")
       if self.Kinetic[0][1] == 0:
           Rconn.r.par = self.Kinetic[0]
       if self.Kinetic[0][2] == 0:
           Rconn.r.par = self.Kinetic[0][:1]
       else:
           Rconn.r.par = self.Kinetic
       
       print pathname
       
       Rconn.r.irfpar = self.IRF_para
       
       print "1"
       
       Rconn("""invisible(mdDat <- readData('we_data_file.txt'))
           gtaModel1 <- initModel(mod_type = "kin",kinpar = par,irfpar = irfpar,fixed = list(irfpar=c(1),clpequ=1:0))
       """)
       
       results = Rconn("gtaFitResult <- fitModel(data = list(mdDat),modspec = list(gtaModel1),modeldiffs = list(linkclp = list(c(1))),opt = kinopt(iter = 10, stderrclp = TRUE, kinspecerr = TRUE, plot = FALSE)")
       print "4"
       self.Results = results
