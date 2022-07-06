# A probabilistic classification of X-ray sources
## Classification of X-ray sources using Bayesian Optimized Inference (CLAXBOI)

This folder contains all the necessary to run the CLAXBOI code (Tranin et al. 2021 submitted to A&A) once your X-ray catalogue is augmented as follows:
1. first column is the identifier
2. there is a column named "class", it is NULL for unknown objects and contains the class label for the reference sample
3. other columns contain all properties you want to use (and more if needed)

### Usage:

First of all, please update the file <code>configfile.ini</code> with your filenames and data-dependent parameters.
Then you can run

<code>python3 classify_new.py</code>

(follow the instructions)

Only after launching the previous command you can set <code>optimize_coeffs: 1</code> in <code>configfile.ini</code>, to perform the optimization if you wish to. This may take a few hours, but you may interrupt it before it converges (Ctrl+C).
Then run again

<code>python3 classify_new.py</code>

to optimize the classification on the class of your choice. Finally, set <code>save: 1</code> and <code>optimize_coeffs: 0</code>
in <code>configfile.ini</code> and run <code>python3 classify_new.py</code> a last time.

Input distributions and density estimations will be automatically plotted in the directory <code>dirref</code>.

<code>python3 plotdistrib.py</code>

Eventually, to check the probability track of a particular source:

<code>python3 Pbatrack.py *SourceIdentifier*</code>


