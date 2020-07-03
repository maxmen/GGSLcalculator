# GGSLcalculator

Script to calculate the GGSL cross section as a function of source redshift for arbitrary lenses, released as documentation for the Science publication Meneghetti et al., 2020.

The code is written in python and depends on few publicly available packages:

* astropy
* numpy
* scipy
* shapely
* scikit-image

## Instruction to run the code

To run the script, just type

    python GGSL_cs_computation.py --input <input file.fits>
    
The input file must be in fits format. It contains the maps of the two components of the lens deflection angle. The two maps are stored in separe HDUs. Lens and source redshifts must be changed in the script according to the input deflection angle map (keywords zl and zs).

An example of input file is provided with the package (`alpha.fits`).
