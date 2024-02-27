# A probabilistic classification of X-ray sources
## Classification of X-ray sources using Naive Bayes Optimized Inference (CLAXBOI)

This folder contains all the necessary to run the CLAXBOI code (Tranin et al. 2022, A&A 657, 138) to augment and classify your X-ray catalog.
Requirements of the system:
1. Python >3.6
2. Ubuntu (for a few os.system commands)
3. Commonly-used Python packages in astrophysics (os, sys, numpy, scipy, astropy, yaml, tqdm)
4. NWAY (https://github.com/JohannesBuchner/nway) must be installed on your machine. Alternatively, you can skip the multiwavelength crossmatching step, but in that case it is preferable to have identified multiwavelength counterparts beforehand in your input catalog.

Requirements of the catalog:
1. first column is the identifier
2. there are columns for the coordinates (their name must be or contain "ra" and "dec" or "de")
3. there is a column for the X-ray flux

### Usage:

Update the file <code>configfile.ini</code> with your filenames and data-dependent parameters (some default values are already provided). 
Run the following codes to transform your X-ray catalog into a value-added catalog (with multiwavelength and variability information).
1. <code>auto_nway.py</code>: identify multiwavelength counterparts
2. <code>auto_xlinks.py</code>: compute basic (multi-mission) X-ray variability information
3. <code>auto_gaiaglade.py</code>: add Gaia distance and proper motion for matching (Galactic) sources. Add GLADE distance and separation for matching (extragalactic) sources.
4. <code>auto_classes.py</code>: identify a training sample for each class using Vizier and Simbad databases.
5. <code>classify_new.py</code>: run the Naive Bayes Classification for probabilistic source identification in the whole X-ray catalog. Optimize this classification.


### Documentation on <code>auto_nway.py</code>
NWAY is a Bayesian probabilistic cross-matching tool able to identify multiwavelength counterparts in large astronomical catalogs. It is used here in its simplest mode of correlating one primary catalog (the X-ray catalog) to another secondary catalog, based on geometrical (angular separations) and physical (magnitude priors) information.

<code>auto_nway.py</code> is used to identify multiwavelength counterparts of your X-ray sources. By default, the following set of wavedomain and CDS catalogues are used to search for counterparts:
* Optical: Gaia EDR3, PanSTARRS DR1, DES DR1, USNO-B1.0
* Infrared: 2MASS, AllWISE, UnWISE
This choice is motivated by the complementarity and combined completeness of these surveys. You can complement or update this set of catalogs by changing the <code>cds_table</code> list in <code>auto_nway.py</code> (and changing the <code>cds_cov</code>, <code>cds_ntot</code>, <code>cds_names</code> lists accordingly).
The maximum angular separation is set by the <code>radius</code> parameter (default value: 8 arcsec).
Since NWAY works on local files, eligible counterpart candidates are first searched and downloaded in cones of radius <code>1.25*radius</code> around each X-ray (true or fake) coordinates.

To obtain a good purity-completeness trade-off, NWAY features a code to calibrate a cutoff on the <code>p_any</code> quantity (i.e. probability that the source in the primary catalog has any counterpart in the secondary catalog). X-ray to optical (resp. infrared) associations having a <code>p_any</code> lower than this threshold are considered non-matches. This code is based on a comparison between matches with true and fake coordinates. Therefore, <code>auto_nway.py</code> generates fake coordinates for each X-ray source (random offset of up to 2 degrees in RA and DEC values) and run the NWAY calibration code, which computes the p_any threshold corresponding to different false positive rates. You can tune the false positive rate you need (default value 15%) by changing the <code>nwaycutoffs</code> list (the last value of the list is used as final calibration).

For each pair of X-ray / multiwavelength catalog, a file is saved containing the candidate associations along with their NWAY probabilities.
Each X-ray source is then assigned a single (or no) optical (resp. infrared) counterpart, i.e. the best-matching reasonable counterpart (highest <code>p_i</code> and <code>p_any>threshold</code>) in the first multiwavelength catalog in which it was found (thus the order of <code>cds_tables</code> _does_ count).

The output of the code if a catalog "..._with_counterparts.fits" containing the same number of entries as the input catalog.

### Documentation on <code>auto_xlinks.py</code>

This program identifies common sources in different X-ray catalogues, assigning a unique MASTER_ID to each unique source, and computes the maximum-to-miminum flux ratios (with or without flux error). A special attention is given to the identification and removal of any ambiguous match, where a source is associated to several other sources at both 1 and 3 sigma (position errors). Spurious and flagged detections may be discarded as well.

Before running it, you need to change the variables <code>xmmdet</code>, <code>chadet</code> and <code>swidet</code> to match your local XMM, Chandra and Swift catalogs of detections.

The output of the code if a catalog "..._with_counterparts_x.fits" containing the same number of entries as the input catalog.

For more advanced X-ray-to-X-ray matches, please use the STONKS pipeline (https://github.com/ErwanQuintin/STONKS).

### Documentation on <code>auto_gaiaglade.py</code>

<code>auto_gaiaglade.py</code> augments the X-ray catalog by adding information about the source distance and motion.

Proper motions and Gaia-based distances (from Bailer-Jones et al. 2021) are added to those sources having a Gaia counterpart identified in the previous step. The X-ray mean luminosity based on this distance is then computed. This luminosity <code>Lx_2</code> should not be used for extragalactic sources.

Extragalactic sources are searched in nearby galaxies using the GLADE (2016 version, Dalya et al. 2018) catalog of galaxies, by performing an ellipse crossmatch using the D25 ellipse representing the visible area of the galaxy (or more exactly the ellipse of 1.26 times this radius, matching the Holmberg radius). The distance to the galaxy is assigned to X-ray matches to compute another luminosity (<code>Lx_1</code>). Another useful quantity is <code>SepToRadius</code>, the galactocentric radius, representing the ratio of the source angular separation (to the galaxy nucleus) to the radius of the galaxy at the source's position angle. <code>Lx_1</code> should not be used for Galactic sources.

At this stage, the quantities <code>logFxFb, logFxFr</code> and <code>logFxFw1, logFxFw2</code> are computed, representing the logarithm of the ratio between X-ray and optical (Gaia BP and RP bands) fluxes, and the ratio between X-ray and infrared (WISE W1 and W2 bands) fluxes, respectively.

The output of the code if a catalog "..._with_counterparts_x_loc.fits" containing the same number of entries as the input catalog.

### Documentation on <code>auto_classes.py</code>

This code is used to build the training sample for each class. The present version of CLAXBOI is based on 7 classes:
* (distant) AGN and QSO
* Star
* Galactic X-ray binary
* Galactic CV
* Nearby AGN (or background QSO behind nearby galaxies)
* Extragalactic X-ray binary
* Extended source

Known sources of the first six types are identified using Vizier catalogs of AGN, stars, XRB and CV. You can find (and update) the dictionary of Vizier catalogs in the <code>vizcat</code> variable.
The difference between nearby and distant AGN (resp. Galactic and extragalactic XRB) is the presence or absence of GLADE-association data.

As a result, columns <code>isAGN, isStar, isXRB</code> and <code>isCV</code> are added to the X-ray catalog, containing a bitwise flag encoding the reference(s) where the identification was (were) found. Example for <code>isAGN</code>: 1 means it is identified as AGN given its mid-infrared emission (Secrest+2015), 2 means it is in the Veron-Cetty & Veron catalog, and 3 means it is in both catalogs.

When a source is unambiguously identified in the set of Vizier catalogs (meaning one of the <code>is...</code> column is non-zero and all other are zero), the value encoding the source type (which is by default, 0: QSO, 1: Star, 2: gal_XRB, 3: CV, 4: AGN, 5: ex_xrb, 6: extended) is stored in the column <code>class</code>, later used for training.


### Documentation on <code>classify_new.py</code>

If you are not using the default CLAXBOI classes, you must update the <code>classnames</code> and <code>trueprop</code> variables in <code>configfile.ini</code>. The variable <code>trueprop</code> is a list of user-informed proportions of each class, expected to represent the underlying proportion of each class in the whole catalog (the sum must be one).

When you run 
<code>python3 classify_new.py</code>, instructions are given to help you fill missing data before running the Naive Bayes Classification. These data are (or will be after first run) contained in the files <code>configfile.ini</code> and <code>[name of the input catalog].in</code>.

Only after launching the previous command you can set <code>optimize_coeffs: 1</code> in <code>configfile.ini</code>, to perform the optimization if you wish to. This may take a few hours, but you may interrupt it before it (hopefully) converges (Ctrl+C).
Then run again

<code>python3 classify_new.py</code>

to optimize the classification on the class of your choice. Finally, set <code>save: 1</code> and <code>optimize_coeffs: 0</code>
in <code>configfile.ini</code> and run <code>python3 classify_new.py</code> one last time.

Input distributions and density estimations will be automatically plotted and saved to the directory <code>dirref</code>.

<code>python3 plotdistrib.py</code> can be used to plot these distributions again.

Eventually, to check the probability track (main interpretability tool) of a particular source:

<code>python3 Pbatrack.py *SourceIdentifier*</code>
