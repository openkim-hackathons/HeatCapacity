Heat Capacity and Thermal Expansion Test Driver
===============================================

Test driver that estimates the constant-pressure heat capacity and linear thermal expansion tensor with
finite-difference numerical derivatives.

Section 3.2 in https://pubs.acs.org/doi/10.1021/jp909762j argues that the centered finite-difference approach
is more accurate than fluctuation-based approaches for computing heat capacities from molecular-dynamics
simulations.

The finite-difference approach requires to run at least three molecular-dynamics simulations with a fixed number of
atoms N at constant pressure (P) and at different constant temperatures T (NPT ensemble), one at the target temperature
at which the heat capacity and thermal expansion tensor are to be estimated, one at a slightly lower temperature, and
one at a slightly higher temperature. It is possible to add more temperature points symmetrically around the target
temperature for higher-order finite-difference schemes.

This test driver repeats the unit cell of the zero-temperature crystal structure to build a supercell and then runs
molecular-dynamics simulations in the NPT ensemble using Lammps.

This test driver uses kim_convergence to detect equilibrated molecular-dynamics simulations. It checks for the
convergence of the volume, temperature, enthalpy and cell shape parameters every 10000 timesteps.

During each equilibrated part of the simulations, the test driver averages the cell parameters and atomic positions to
obtain the equilibrium crystal structures. This includes an average over time, and an average over the replicated unit
cells.

After the molecular-dynamics simulations, the symmetry of the average structures are checked to ensure that it did not
change in comparison to the initial structure. Also, it is ensured that replicated atoms in replicated unit atoms are
not too far away from the average atomic positions.

The crystals might melt or vaporize during the simulations. In that case, kim-convergence would only detect
equilibration after unnecessarily long simulations. Therefore, this test driver initially check for melting or
vaporization during short initial simulations. During these initial runs, the mean-squared displacement (MSD) of atoms
during the simulations is monitored. If the MSD exceeds a given threshold value, an error is raised.
