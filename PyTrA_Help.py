from enthought.traits.ui.api import *
from enthought.traits.api import *

class Help(HasTraits):
    text = ("""
<html><body>

<h1>PyTrA Help</h1>
    
<p><a href="http://www.youtube.com/watch?v=kcGj8FnVCDI">Click here to see the video</a>

<p>More information on the website<a href="nznano.blogspot.co.nz">nznano</a>

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

<p>You may want to delete a range of wavelengths between which the data is offscale click the <strong>Delete traces</strong> button and pick the wavelengths you want to delete between</p>

<h3>Correcting Chirp</h3>

<p>Due to the ultrashort pulses from femtosecond transient absorption spectrometers shorter wavelengths arrive at the spectrometer before longer wavelengths</p>

<p>To correct for this we can fit a polynomial to the chirp spectrum and then interpolate our values to correct for the dispersion</p>

<p>View the chirp by clicking the <strong>2D plot of chirp</strong> button</p>

<p>Now find the times between which the chirp is visible.</p>

<p>Close the plot and input the time range when the chirp is visible in the two text boxes under the <strong>Time range for chirp correction</strong> label</p>

<p>Click the <strong>Fix for chirp</strong> button. You will be asked to pick 8 points on the chirp wavefront and then another graph will ask you to pick the top of the wavefront</p> 

<p>The polynomial fit and the standard deviation and added to the bottom text bar</p>

<h3>Data anaylsis</h3>

<p>To fit a single trace at a certain wavelength click the <strong>Fit trace</strong> button plot will appear click on a wavelength that you want to fit to</p>

<p>The pymodelfit window will open and the data will be displayed the trace.</p>

<img src="fitting.jpg"/>

<p> In fitting the data it is recommended that you click the <strong>Fit Model</strong> button until it converges and the solution no longer changes</p>

<p>On exit the parameters used to fit the trace are printed in the status bar</p>

<p>Pick the model that you want to use click on the new model button. 
<p>Convoluted_1 is a single exponential with the instrument response (gaussian)</p>
<p>Convoluted_2 is a double expoential convoluted with the instrument response (gaussian)</p>
<p>Click the Fit model multiple times until the variable do not change. Each time you fit the model the algorithim uses the varibles from the previous fit to get a better intial guess. This leads to an accurate fit of the data and leads to what is called convergence.</p>
<p>The variables that are used in the fit.</p>
<p>A1 = Amplitude</p>
<p>mu = shift in the centre of the gaussian distribution</p>
<p>w = Full width half maximum of the gaussian (instrument response)</p>
<p>T1 = Time constant</p>
<p>y0 = baseline</p>
<p>Where w gives us the width of the instrument at half the height of the peak and T1 gives us the time constant for the exponential decay.</p>

<h3>Global fitting</h3>

Global fitting can be done using Igor Pro from Wavemetrics. Otherwise TIMP library for R is a free alternative.

<p>Click the <strong>Select multiple traces</strong> button. You will be prompted to pick between which wavelengths to fit to.</p>

<p>The number of traces saved and between which wavelengths will be printed in the Status bar</p>

<p>Make sure Igor Pro is open and then click on the <strong>Send traces to Igor</strong> Igor Pro should prompt you to load the data click the <strong>Load</strong> button the traces will be inputted and the time vector will be saved as timevals</p>

<p>The Global Analysis window wil open click on <strong>Add/Remove Waves</strong></p>

<p>Select wave1 to wave(end) for the Y Waves and timeval as the X waves. Click the middle arrow to populate the list then click <strong>ok</strong></p>

<p>Click on the Function column title to select all of the traces Function. Click <strong>Choose Fit Function</strong> and choose the function you want to fit.</p>

<p>Link all the the coefficients by clicking on the <strong>Select Coef Column</strong> button and select the first column click <strong>Link Selection</strong>. Repeat for all the coefficents so that all of them are coloured. This makes inputing the data easier.</p>

<p>Click on the <strong>Coefficient Control</strong> tab. Enter the initial guess from the PyTrA single wavelength fit. Return to the <strong>Data Sets and Functions</strong> tab and select the columns in the same way as earlier and unlink the selection. Leaving only the time constants columns linked.</p>

<p>Click <strong>Fit!</strong> and wait for Igor Pro to compute the global minimum. The coefficients will be printed out make sure in the output it reads Global Fit converged normally. Note down the global time constant which is the same for all of the data traces.</p>

<h3>Visualisation</h3>

<p>A 3D plot can be opened by clicking the <strong>3D plot</strong> button. The surface will be sent to MayaVi window for viewing. You can customise the plot using the tool bar on the top of the 3D plot the far left button opens up a dialog with many different options for label sizes titles and surface colours.</p>

<p>2D contour plot, spectra at a certain time value and trace at a certain wavelength can be viewed using matplotlib.</p> 

<h3>Exporting data</h3>

<p>Data can be exported as csv in same format it was imported as. The file is saved in the same directory where the data was taken and starts with the date in front with the name of the file taken from the title.</p>

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