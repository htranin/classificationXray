# A probabilistic classification of X-ray sources
## Classification of X-ray sources using Bayesian Optimized Inference (CLAXBOI)

This folder contains all the necessary to run the CLAXBOI code (Tranin et al. 2021) once your X-ray catalogue is augmented as follows:
1. first column is the identifier
2. there is a column named "class", it is NULL for unknown objects and contains the class label for the reference sample
3. other columns contain all properties you want to use (and more if needed)

### Usage:

<code>python3 classify_new.py *NameOfYourCatalogue*.csv</code>

(follow the instructions)

Only after launching the previous command you can run:

<code>python3 classify_new.py --optimize 2</code>

to optimize the classification on class labelled as 2.
**Update 20/02/21: "optimize" option is not yet implemented. Coming soon...**

To plot the distributions computed by the first command, try:

<code>python3 plotdistrib.py</code>

Eventually, to check the probability track of a particular source:

<code>python3 Pbatrack.py *SourceIdentifier*</code>


