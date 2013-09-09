from enthought.traits.ui.api import *
from enthought.traits.api import *

class Help(HasTraits):
    text = ("""
<html><body>

<h1>PyTrA Help</h1>
    
<p><a href="http://www.youtube.com/watch?v=kcGj8FnVCDI">Click here to go to the website</a>

<h3>Importing Data</h3>

<p>The data can be in csv or text delimited format.</p> 

<table>
<tr>
  <td>0</td>
  <td>time(1)</td>
  <td>time(2)</td>
  <td>...</td>
  <td>time(m)</td>
</tr>
<tr>
  <td>lambda(1)</td>
  <td>abs(1,1)</td>
  <td>abs(1,2)</td>
  <td>...</td>
  <td>abs(1,m)</td>
</tr>
<tr>
  <td>lambda(2)</td>
  <td>abs(2,1)</td>
  <td>abs(2,2)</td>
  <td>...</td>
  <td>abs(2,m)</td>
</tr>
<tr>
  <td>:</td>
  <td>:</td>
  <td>:</td>
  <td>:</td>
  <td>:</td>
</tr>
<tr>
  <td>lambda(n)</td>
  <td>abs(n,1)</td>
  <td>abs(n,2)</td>
  <td>...</td>
  <td>abs(n,m)</td>
</tr>
</table>

<p>To select the data click on the <strong>folder</strong> icon and select the TrA data file you want to import and the chirp spectrum you want to import. Click on the <strong>Load data</strong> button to import the data</p>

<p>Importing data from Ohio State University involves inputing the data file and the delay file (A single vector of the time delays). The wavelength and corresponding pixel number are input for calibrating the data. To load the data press <strong>Load data</strong> button and click <strong>ok</strong></p>

<h3>Correcting Chirp</h3>

<p>Due to the ultrashort pulses from femtosecond transient absorption spectrometers shorter wavelengths arrive at the spectrometer before longer wavelengths</p>

<p>To correct for this we can fit a polynomial to the chirp spectrum and then interpolate our values to correct for the dispersion</p>

<p>View the chirp by going to the Chirp correction menu and click on Plot chirp</p>

<p>Zoom into the area you want to fit the chirp to</p>

<p>Click the Chirp correction item in the Chirp correction menu. You will be asked to pick 8 points on the chirp wavefront and then another graph will ask you to pick the front of the wavefront</p>

<p>The polynomial fit and the standard deviation are printed in the log</p>

<h3>Data anaylsis</h3>

<p>To fit a single trace or spectra go to type the value into the text box under the line plots then click either <strong>Fit Trace</strong></p>

<p>The pymodelfit window will open and the data will be displayed.</p>

<p><img src="fitting.jpg"/></p>

<p>On exit the parameters used to fit the trace are printed in the status bar</p>

<p>Pick the model that you want to use click on the new model button. 
<p>Convoluted_1,2,3,4 are exponential sums convoluted with the instrument response (gaussian)</p>
<p>Gauss_1,2,3,4 are Gaussian function with x and y shifts</p>
<p>Click the Fit model multiple times until the variable do not change. Each time you fit the model the algorithim uses the varibles from the previous fit to get a better intial guess. This leads to an accurate fit of the data where the solution convergence on the best value for the model and data.</p>

<p>The variables that are used in the fit.</p>
<p>A = Amplitude</p>
<p>mu = shift in the centre of the gaussian distribution</p>
<p>w = Full width half maximum of the gaussian (instrument response)</p>
<p>T = Time constant</p>
<p>y0 = baseline</p>

<p>Where w gives the width of the instrument at half the height of the peak and T gives us the time constant for the exponential decay.</p>

<h3>Global fitting</h3>

Global fitting can be done using Igor Pro from Wavemetrics. Otherwise TIMP library for R is a free alternative.

<p>Click the <strong>Select traces</strong> button under the Global fit menu item. You will be prompted to pick between which wavelengths to fit to.</p>

<p>The number of traces saved and between which wavelengths will be printed in the log file</p>

<p>Make sure Igor Pro is open and then click on the <strong>Send to Igor</strong> in the Global fit menu item. Igor Pro should prompt you to load the data click the <strong>Load</strong> button the traces will be inputted and the time vector will be saved as timevals</p>

<p>The Global Analysis window wil open click on <strong>Add/Remove Waves</strong></p>

<p>Select wave1 to wave(end) for the Y Waves and timeval as the X waves. Click the middle arrow to populate the list then click <strong>ok</strong></p>

<p>Click on the Function column title to select all of the traces Function. Click <strong>Choose Fit Function</strong> and choose the function you want to fit.</p>

<p>Link all the the coefficients by clicking on the <strong>Select Coef Column</strong> button and select the first column click <strong>Link Selection</strong>. Repeat for all the coefficents so that all of them are coloured. This makes inputing the data easier.</p>

<p>Click on the <strong>Coefficient Control</strong> tab. Enter the initial guess from the PyTrA single wavelength fit. Return to the <strong>Data Sets and Functions</strong> tab and select the columns in the same way as earlier and unlink the selection. Leaving only the time constants columns linked.</p>

<p>Click <strong>Fit!</strong> and wait for Igor Pro to compute the global minimum. The coefficients will be printed out make sure in the output it reads Global Fit converged normally. Note down the global time constant which is the same for all of the data traces.</p>

<h3>MCMC</h3>

<p>One method for model checking and non-linear regression is to use Markov Chain Monte Carlo. This method defines all of the parameters and the error as distribution from which to sample from, these are the priors. You then can sample the distributions to see if a stable solution exist and if the chain converges</p>

<p>To begin you need to do a single trace fit of the kinetics and open up the MCMC dialog box found under the kinetic trace in the main window</p>

<p><img src="mcmc.jpg"/></p>

<p>Setting the parameters you will need to choose priors that do not restrict the parameters but hold the parameter from becoming unphysical (such as a negative time decay tau). It is recomended to only use Uniform priors</p>

<p>Once the chain has run you should see the plots of each parameter. The top left plot is of the chain value at the different iterations, the chain must converge to a stable value. If converge does not occur the model is not a stable solution within the noise of the system. The bottom left graph is the autocorrelation plot. This shows the amount of correlation of the sample with the samples previously, we want a Markov chain which is only correlated to the previous state this being an autocorrelation plot showing minimal correlation. The right plot is the histogram for the distribution of the chain we want a high density around a mean.</p>

<p><img src="chain.jpg"/></p>

<p>The next step is tuning the chain. If the chain has converged you can use the Raftery Lewis statistics to set the values for the burn-in, thining and number of iterations.</p>

<p> Once you are finished shut the window the values of the mean and standard deviation will be printed out in the log file.</p>

<p>The Bayesian information Criterion (BIC) can be used to compare different models. This is calculated from a maximum a posteriori (MAP) which optimises the parameters to the value of maximum denisty. More negative BIC indicate more dense distribution and a better fit with the number of parameters also taken into account.</p>

<h3>Visualisation</h3>

<p>A 3D plot can be opened by clicking the <strong>3D plot</strong> button in the 2d graph tab. The surface will be sent to MayaVi window for viewing. You can customise the plot using the tool bar on the top of the 3D plot the far left button opens up a dialog with many different options for label sizes titles and surface colours. By changing the value right of the 3D plot button you can scale the z axis of the plot</p>

<p>The plots in the main window can be zoomed by holding Ctrl and dragging over the area of interest. The <strong>Averaged</strong> and <strong>Normalised</strong> buttons will slice the 2D plot into 10 sections within the zoomed area of the main 2D plot in the main window. SVD and EFA also work within the zoom range of the main 2D plot.</p>

<p>2D contour plot, spectra at a certain time value and trace at a certain wavelength can be viewed using matplotlib as well which can produce publication quality images.</p>

<h3>Exporting data</h3>

<p>Data can be exported as csv in same format it was imported as. The file is saved in the same directory where the data was taken and starts with the date in front with the name of the file taken from the title. The log file will also be saved in the same place if the <strong>Save log file</strong> is clicked.</p>
</body>
</html> 
""")
    html = HTML(text, editor=HTMLEditor(open_externally=True))
    
    view = View(
        Item('html', show_label=False),
        title     = 'PyTrA Help',
        buttons   = ['OK'],
        width     = 800,
        height    = 600,
        resizable = True)