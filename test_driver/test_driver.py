from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import random
import shutil
from typing import Optional, Sequence
from ase.calculators.lammps import convert, Prism
import numpy as np
from kim_tools import get_stoich_reduced_list_from_prototype, KIMTestDriverError
from kim_tools.symmetry_util.core import (reduce_and_avg, PeriodExtensionException,
                                          fit_voigt_tensor_to_cell_and_space_group)
from kim_tools.test_driver import SingleCrystalTestDriver
from .helper_functions import (compute_alpha_tensor, compute_heat_capacity, get_cell_from_averaged_lammps_dump,
                               get_positions_from_averaged_lammps_dump, run_lammps)


class TestDriver(SingleCrystalTestDriver):
    def _calculate(self, temperature_step_fraction: float = 0.01, number_symmetric_temperature_steps: int = 1,
                   timestep_ps: float = 0.001, number_sampling_timesteps: int = 100,
                   repeat: Optional[Sequence[int]] = None, max_workers: Optional[int] = None,
                   lammps_command: str = "lmp", msd_threshold_angstrom_squared_per_sampling_timesteps: float = 0.1,
                   number_msd_timesteps: int = 20000, random_seeds: Optional[Sequence[int]] = (1, 2, 3),
                   rlc_n_every: int = 10, rlc_run_length: int = 10000, rlc_min_samples: int = 5,
                   output_dir: str = "output", equilibration_plots: bool = True, **kwargs) -> None:
        """
        Estimate constant-pressure heat capacity and linear thermal expansion tensor with finite-difference numerical
        derivatives.

        Section 3.2 in https://pubs.acs.org/doi/10.1021/jp909762j argues that the centered finite-difference approach
        is more accurate than fluctuation-based approaches for computing heat capacities from molecular-dynamics
        simulations.

        The finite-difference approach requires to run at least three molecular-dynamics simulations with a fixed number
        of atoms N at constant pressure (P) and at different constant temperatures T (NPT ensemble), one at the target
        temperature at which the heat capacity and thermal expansion tensor are to be estimated, one at a slightly lower
        temperature, and one at a slightly higher temperature. It is possible to add more temperature points
        symmetrically around the target temperature for higher-order finite-difference schemes.

        This test driver repeats the unit cell of the zero-temperature crystal structure to build a supercell and then
        runs molecular-dynamics simulations in the NPT ensemble using Lammps.

        This test driver uses kim_convergence to detect equilibrated molecular-dynamics simulations. It checks for the
        convergence of the volume, temperature, enthalpy and cell shape parameters every 10000 timesteps.

        After the molecular-dynamics simulations, the symmetry of the average structures during the equilbrated parts of
        the runs are checked to ensure that they did not change in comparison to the initial structure. Also, it is
        ensured that replicated atoms in replicated unit atoms are not too far away from the average atomic positions.

        The crystals might melt or vaporize during the simulations. In that case, kim-convergence would only detect
        equilibration after unnecessarily long simulations. Therefore, this test driver initially check for melting or
        vaporization during short initial simulations. During these initial runs, the mean-squared displacement (MSD) of
        atoms during the simulations is monitored. If the MSD exceeds a given threshold value, an error is raised.

        All output files are written to the given output directory.

        :param temperature_step_fraction:
            Fraction of the target temperature that is used as temperature step for the finite-difference scheme.
            For example, if the target temperature is 300 K and the temperature_step_fraction is 0.1, the temperature
            difference between the different NPT simulations will be 30 K.
            Should be bigger than zero and smaller than one divided by number_symmetric_temperature_steps (to avoid
            simulations at negative temperatures).
            Default is 0.01 (1% of the target temperature).
            Should be bigger than zero and smaller than one.
        :type temperature_step_fraction: float
        :param number_symmetric_temperature_steps:
            Number of symmetric temperature steps around the target temperature to use for the finite-difference
            scheme.
            For example, if number_symmetric_temperature_steps is 2, five NPT simulations will be run at temperatures
            T - 2*delta_T, T - delta_T, T, T + delta_T, T + 2*delta_T, where delta_T is determined by
            temperature_step_fraction * T.
            Default is 1.
            Should be bigger than zero.
        :type number_symmetric_temperature_steps: int
        :param timestep_ps:
            Time step in picoseconds.
            Default is 0.001 ps (1 fs).
            Should be bigger than zero.
        :type timestep_ps: float
        :param number_sampling_timesteps:
            Sample thermodynamic variables every number_sampling_timesteps timesteps in Lammps.
            Default is 100 timesteps.
            Should be bigger than zero.
        :type number_sampling_timesteps: int
        :param repeat:
            Tuple of three integers specifying how often to repeat the unit cell in each direction to build the
            supercell.
            If None, a supercell size close to 10,000 atoms is chosen.
            Default is None.
            If not None, all entries have to be bigger than zero.
        :type repeat: Sequence[int]
        :param max_workers:
            Maximum number of parallel workers to use for running Lammps simulations at different temperatures.
            If None is given, this will be set to 1.
            This is independent of the number of processors used by each Lammps simulation that can be specified in the
            lammps command itself.
            Default is None.
        :type max_workers: Optional[int]
        :param lammps_command:
            Command to run Lammps.
            Default is "lmp".
        :type lammps_command: str
        :param msd_threshold_angstrom_squared_per_sampling_timesteps:
            Mean-squared displacement threshold in Angstroms^2 per number_sampling_timesteps to detect melting or
            vaporization.
            Default is 0.1.
            Should be bigger than zero.
        :type msd_threshold_angstrom_squared_per_sampling_timesteps: float
        :param number_msd_timesteps:
            Number of timesteps to monitor the mean-squared displacement in Lammps.
            Before the mean-squared displacement is monitored, the system will be equilibrated for the same number of
            timesteps.
            Default is 20000 timesteps.
            Should be bigger than zero and a multiple of number_sampling_timesteps.
        :param random_seeds:
            Random seeds for the Lammps simulations.
            This has to be a sequence of 2 * number_symmetric_temperature_steps + 1 integers for the different
            temperatures being simulated.
            If None is given, random seeds will be sampled.
            Default is (1, 2, 3).
            Each seed should be bigger than zero.
        :type random_seeds: Optional[Sequence[int]]
        :param rlc_n_every:
            Number of timesteps between storage of values for the run-length control in kim-convergence.
            Default is 10.
            Should be bigger than zero.
        :type rlc_n_every: int
        :param rlc_run_length:
            Run length in timesteps for run-length control with kim-convergence.
            This will also be the timestep interval in generated trajectory files.
            Default is 10000 timesteps.
            Should be bigger than zero and a multiple of number_sampling_timesteps.
        :type rlc_run_length: int
        :param rlc_min_samples:
            Minimum number of independent samples for convergence in run-length control with kim-convergence.
            Default is 5.
            Should be bigger than zero.
        :type rlc_min_samples: int
        :param output_dir:
            Directory to which all output files will be written.
            Default is "output".
        :type output_dir: str
        :param equilibration_plots:
            Whether to generate diagnostic plots for the equilibration checks in kim-convergence.
            Default is True.
        :type equilibration_plots: bool

        :raises ValueError:
            If any of the input arguments are invalid.
        :raises KIMTestDriverError:
            If the crystal melts or vaporizes during the simulation.
            If the symmetry of the structure changes.
            If the output directory does not exist.
        """
        # Set prototype label.
        self.prototype_label = self._get_nominal_crystal_structure_npt()["prototype-label"]["source-value"]

        # Get temperature in Kelvin.
        temperature_K = self._get_temperature(unit="K")

        # Get cauchy stress tensor in bar.
        cell_cauchy_stress_bar = self._get_cell_cauchy_stress(unit="bar")

        # Check arguments.
        if not temperature_K > 0.0:
            raise ValueError("Temperature has to be larger than zero.")

        if not len(cell_cauchy_stress_bar) == 6:
            raise ValueError("Specify all six (x, y, z, xy, xz, yz) entries of the cauchy stress tensor.")

        if not (cell_cauchy_stress_bar[0] == cell_cauchy_stress_bar[1] == cell_cauchy_stress_bar[2]):
            raise ValueError("The diagonal entries of the stress tensor have to be equal so that a hydrostatic "
                             "pressure is used.")

        if not (cell_cauchy_stress_bar[3] == cell_cauchy_stress_bar[4] == cell_cauchy_stress_bar[5] == 0.0):
            raise ValueError("The off-diagonal entries of the stress tensor have to be zero so that a hydrostatic "
                             "pressure is used.")

        if not timestep_ps > 0.0:
            raise ValueError("Timestep has to be larger than zero.")

        if not 0.0 < temperature_step_fraction < 1.0:
            raise ValueError("Temperature-step fraction has to be bigger than zero and smaller than one.")

        if not number_symmetric_temperature_steps > 0:
            raise ValueError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise ValueError(
                "The given number of symmetric temperature steps and the given temperature-step fraction "
                "would yield zero or negative temperatures.")

        if not number_sampling_timesteps > 0:
            raise ValueError("Number of timesteps between sampling in Lammps has to be bigger than zero.")

        if repeat is not None:
            if not len(repeat) == 3:
                raise ValueError("The repeat argument has to be a tuple of three integers.")

            if not all(r > 0 for r in repeat):
                raise ValueError("All number of repeats must be bigger than zero.")

        if max_workers is not None:
            if not max_workers > 0:
                raise ValueError("Maximum number of workers has to be bigger than zero.")
        else:
            max_workers = 1
        
        if not msd_threshold_angstrom_squared_per_sampling_timesteps > 0.0:
            raise ValueError("The mean-squared displacement threshold has to be bigger than zero.")

        if not number_msd_timesteps > 0:
            raise ValueError("The number of timesteps to monitor the mean-squared displacement has to be bigger than "
                             "zero.")

        if not number_msd_timesteps % number_sampling_timesteps == 0:
            raise ValueError("The number of timesteps to monitor the mean-squared displacement has to be a multiple of "
                             "the number of sampling timesteps.")

        if random_seeds is not None:
            if len(random_seeds) != 2 * number_symmetric_temperature_steps + 1:
                raise ValueError("If random seeds are given, their number has to match the number of temperatures "
                                 "being simulated.")
            if not all(rs > 0 for rs in random_seeds):
                raise ValueError("All random seeds must be bigger than zero.")
        else:
            # Get random 31-bit unsigned integers.
            random_seeds = [random.getrandbits(31) for _ in range(2 * number_symmetric_temperature_steps + 1)]

        if not rlc_n_every > 0:
            raise ValueError("The number of timesteps between storage of values for run-length control has to be "
                             "bigger than zero.")

        if not rlc_run_length > 0:
            raise ValueError("The run length for run-length control has to be bigger than zero.")

        if not rlc_run_length % number_sampling_timesteps == 0:
            raise ValueError("The run length for run-length control has to be a multiple of the number of "
                             "sampling timesteps.")

        if not rlc_min_samples > 0:
            raise ValueError("The minimum number of samples to use for convergence checks in run-length control has to "
                             "be bigger than zero.")

        # Get pressure from cauchy stress tensor.
        pressure_bar = -cell_cauchy_stress_bar[0]

         # Copy original atoms so that their information does not get lost.
        original_atoms = self._get_atoms()

        # Create atoms object that will contain the supercell.
        atoms_new = original_atoms.copy()

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        # Build supercell.
        if repeat is None:
            # Get a size close to 10K atoms (shown to give good convergence)
            x = int(np.ceil(np.cbrt(10000 / len(atoms_new))))
            repeat = (x, x, x)

        assert repeat is not None
        atoms_new = atoms_new.repeat(repeat)

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * temperature_K
        temperatures = [temperature_K + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

        # Make sure output directory for all data files exists and copy over necessary files.
        if not os.path.exists(output_dir):
            raise KIMTestDriverError(f"Output directory '{output_dir}' does not exist.")
        test_driver_directory = os.path.dirname(os.path.realpath(__file__))
        shutil.copyfile(os.path.join(test_driver_directory, "npt.lammps"), f"{output_dir}/npt.lammps")
        shutil.copyfile(os.path.join(test_driver_directory, "run_length_control.py"),
                        f"{output_dir}/run_length_control.py")
        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        with open(f"{output_dir}/accuracies.py", "w") as file:
            print("""from typing import Optional, Sequence

# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
# For cells, we can only use a relative accuracy for all non-zero variables.
# The last three variables, however, correspond to the tilt factors of the orthogonal cell (see npt.lammps which are
# expected to fluctuate around zero. For these, we should use an absolute accuracy instead.""", file=file)
            relative_accuracies = ["0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01", "0.01"]
            absolute_accuracies = ["None", "None", "None", "None", "None", "None", "None", "None", "None"]
            _, _, _, xy, xz, yz = convert(Prism(atoms_new.get_cell()).get_lammps_prism(), "distance",
                                          "ASE", "metal")
            if abs(xy) < 1.0e-6:
                relative_accuracies[6] = "None"
                absolute_accuracies[6] = "0.01"
            if abs(xz) < 1.0e-6:
                relative_accuracies[7] = "None"
                absolute_accuracies[7] = "0.01"
            if abs(yz) < 1.0e-6:
                relative_accuracies[8] = "None"
                absolute_accuracies[8] = "0.01"
            print(f"RELATIVE_ACCURACY: Sequence[Optional[float]] = [{', '.join(relative_accuracies)}]", file=file)
            print(f"ABSOLUTE_ACCURACY: Sequence[Optional[float]] = [{', '.join(absolute_accuracies)}]", file=file)

        with open(f"{output_dir}/rlc_parameters.py", "w") as file:
            print(f"""INITIAL_RUN_LENGTH: int = {rlc_run_length}
MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES: Optional[int] = {rlc_min_samples}""", file=file)

        # Write lammps file.
        structure_file = f"{output_dir}/zero_temperature_crystal.lmp"
        atom_style = self._get_supported_lammps_atom_style()
        atoms_new.write(structure_file, format="lammps-data", masses=True, units="metal", atom_style=atom_style)

        # Run Lammps simulations in parallel.
        assert len(temperatures) == len(random_seeds)
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, (t, rs) in enumerate(zip(temperatures, random_seeds)):
                futures.append(executor.submit(
                    run_lammps, self.kim_model_name, i, t, pressure_bar, timestep_ps, number_sampling_timesteps,
                    species, msd_threshold_angstrom_squared_per_sampling_timesteps, number_msd_timesteps,
                    rlc_run_length, rlc_n_every, output_dir, equilibration_plots, lammps_command, rs))

        # If one simulation fails, cancel all runs.
        for future in as_completed(futures):
            assert future.done()
            exception = future.exception()
            if exception is not None:
                for f in futures:
                    f.cancel()
                raise exception

        # Collect results and check that symmetry is unchanged after all simulations.
        log_filenames = []
        all_cells = []
        middle_temperature_atoms = None
        middle_temperature = None
        for t_index, (future, t) in enumerate(zip(futures, temperatures)):
            assert future.done()
            assert future.exception() is None
            (log_filename, restart_filename, average_position_filename, average_cell_filename,
             melted_crystal_filename) = future.result()
            log_filenames.append(log_filename)

            # Check that crystal did not melt or vaporize.
            with open(log_filename, "r") as f:
                for line in f:
                    if line.startswith("Crystal melted or vaporized"):
                        assert os.path.exists(melted_crystal_filename)
                        raise KIMTestDriverError(f"Crystal melted or vaporized during simulation at temperature {t} K.")
            assert not os.path.exists(melted_crystal_filename)

            # Process results and check that symmetry is unchanged after simulation.
            atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
            atoms_new.set_scaled_positions(
                get_positions_from_averaged_lammps_dump(average_position_filename))
            try:
                reduced_atoms = reduce_and_avg(atoms_new, repeat)
            except PeriodExtensionException as e:
                atoms_new.write(f"{output_dir}/final_configuration_temperature_{t_index}_failing.poscar",
                                format="vasp", sort=True)
                raise KIMTestDriverError(f"Could not reduce structure after NPT simulation at "
                                         f"temperature {t} K (temperature index {t_index}): {e}")

            if t_index == number_symmetric_temperature_steps:
                # Store the atoms of the middle temperature for later because their crystal genome designation 
                # will be used for the heat-capacity and thermal expansion properties.
                middle_temperature_atoms = reduced_atoms.copy()
                middle_temperature = t
            
            # Check that the symmetry of the structure did not change.
            if not self._verify_unchanged_symmetry(reduced_atoms):
                reduced_atoms.write(f"{output_dir}/reduced_atoms_temperature_{t_index}_failing.poscar",
                                    format="vasp", sort=True)
                raise KIMTestDriverError(f"Symmetry of structure changed during simulation at temperature {t} K.")
            
            # Write NPT crystal structures.
            self._update_nominal_parameter_values(reduced_atoms)
            # since we're looping over the futures, one per temperature
            # calling this will append the current cell, one per temperature, 
            # into an array for later use
            all_cells.append(self._get_atoms().cell)
            self._add_property_instance_and_common_crystal_genome_keys("crystal-structure-npt", write_stress=True,
                                                                       write_temp=t)
            self._add_file_to_current_property_instance("restart-file", restart_filename)
            
            # Reset to original atoms.
            self._update_nominal_parameter_values(original_atoms)

        assert middle_temperature_atoms is not None
        assert middle_temperature is not None

        c = compute_heat_capacity(temperatures, log_filenames, 2)
        alpha = compute_alpha_tensor(all_cells, temperatures)

        # Print result.
        print('####################################')
        print('# NPT Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')
        print('####################################')
        print('# NPT Linear Thermal Expansion Tensor Results #')
        print('####################################')
        print(f'alpha:\t{alpha}')

        # Write property.
        max_accuracy = len(temperatures) - 1
        assert len(atoms_new) == len(original_atoms) * repeat[0] * repeat[1] * repeat[2]
        number_atoms = len(atoms_new)
        self._update_nominal_parameter_values(middle_temperature_atoms)
        constant_pressure_heat_capacity = c[f"finite_difference_accuracy_{max_accuracy}"][0]
        constant_pressure_heat_capacity_uncert = c[f"finite_difference_accuracy_{max_accuracy}"][1]
        # If relative uncertainty is too high, print a disclaimer.
        relative_uncertainty = constant_pressure_heat_capacity_uncert / abs(constant_pressure_heat_capacity)
        if relative_uncertainty > 0.1:
            disclaimer = (
                f"The relative uncertainty {relative_uncertainty} of the constant-pressure heat capacity is larger than "
                f"10%. Consider changing the temperature_step_fraction, number_symmetric_temperature_steps and repeat"
                f"arguments to improve results.\nSee stdout and logs for calculation details."
            )
        else:
            disclaimer = None

        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-crystal-npt", write_stress=True, write_temp=middle_temperature, disclaimer=disclaimer)
        self._add_key_to_current_property_instance(
            "heat-capacity-per-atom", constant_pressure_heat_capacity / number_atoms,
            "eV/K",
            uncertainty_info={"source-std-uncert-value": constant_pressure_heat_capacity_uncert / number_atoms})

        number_atoms_in_formula = sum(get_stoich_reduced_list_from_prototype(self.prototype_label))
        assert number_atoms % number_atoms_in_formula == 0
        number_formula = number_atoms // number_atoms_in_formula
        self._add_key_to_current_property_instance(
            "heat-capacity-per-formula", constant_pressure_heat_capacity / number_formula,
            "eV/K",
            uncertainty_info={"source-std-uncert-value": constant_pressure_heat_capacity_uncert / number_formula})

        total_mass_g_per_mol = sum(atoms_new.get_masses())
        self._add_key_to_current_property_instance(
            "specific-heat-capacity", constant_pressure_heat_capacity / total_mass_g_per_mol,
            "eV/K/amu",
            uncertainty_info={"source-std-uncert-value": constant_pressure_heat_capacity_uncert / total_mass_g_per_mol})

        alpha11 = alpha[0][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha22 = alpha[1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha33 = alpha[2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha23 = alpha[3][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha13 = alpha[4][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha12 = alpha[5][f"finite_difference_accuracy_{max_accuracy}"][0]

        # alpha11_err = alpha[0][f"finite_difference_accuracy_{max_accuracy}"][1]
        # alpha22_err = alpha[1][f"finite_difference_accuracy_{max_accuracy}"][1]
        # alpha33_err = alpha[2][f"finite_difference_accuracy_{max_accuracy}"][1]
        # alpha23_err = alpha[3][f"finite_difference_accuracy_{max_accuracy}"][1]
        # alpha13_err = alpha[4][f"finite_difference_accuracy_{max_accuracy}"][1]
        # alpha12_err = alpha[5][f"finite_difference_accuracy_{max_accuracy}"][1]

        # property can be referred to with or without tags
        self._add_property_instance_and_common_crystal_genome_keys("thermal-expansion-coefficient-tensor-npt",
                                                                   write_stress=True, write_temp=True)
        space_group = int(self.prototype_label.split("_")[2])

        # thermal expansion tensor in voigt notation
        alpha_final_voigt_nonsymb = np.asarray([alpha11,alpha22,alpha33,alpha23,alpha13,alpha12])

        # TODO: upgrade to fit_voigt_tensor_and_error_to_cell_and_space_group()
        # once errors are being calculated
        center_cell = all_cells[int(np.floor(len(all_cells)/2))]
        
        alpha_final_voigt_sym = fit_voigt_tensor_to_cell_and_space_group(alpha_final_voigt_nonsymb,
                                                                         center_cell,
                                                                         space_group)
        
        # alpha11 unique for all space groups
        unique_components_names = ["alpha1"]
        unique_components_values = [alpha11]
        # unique_components_errs = [alpha11_err]

        # hexagonal, trigonal, tetragonal space groups alpha33 also unique
        if space_group <= 194:
            unique_components_names.append("alpha3")
            unique_components_values.append(alpha33)
            # unique_components_errs.append(alpha33_err)

        # orthorhombic, alpha22 also unique
        if space_group <= 74:

            # insert alpha22 in the middle so they end up sorted
            # into voigt notation order
            unique_components_names.insert(1,"alpha2")
            unique_components_values.insert(1,alpha22)
            # unique_components_errs.insert(1,alpha22_err)

        # monoclinic or triclinic, all components potentially unique
        if space_group <= 15:

            unique_components_names.append("alpha4")
            unique_components_names.append("alpha5")
            unique_components_names.append("alpha6")

            unique_components_values.append(alpha23)
            unique_components_values.append(alpha13)
            unique_components_values.append(alpha12)

            # unique_components_errs.append(alpha23_err)
            # unique_components_errs.append(alpha13_err)
            # unique_components_errs.append(alpha12_err)

        """
        Presently, errors are not reported because there isn't a good way to get
        the initial uncertainty of the cell parameters. If we determine a good way to do that,
        uncommenting the above lines involving 'unique_components_errs' and 'alphaij_err'
        and replacing the TODO in helper_functions.compute_alpha_voigt() 
        with the initial cell errors should be a drop-in addition, as the code is set up
        to accept and report errors.
        """

        # TODO: add uncertainty info once we decide how to calculate cell errors
        self._add_key_to_current_property_instance("thermal-expansion-voigt-raw", alpha_final_voigt_nonsymb, "1/K")
        self._add_key_to_current_property_instance("thermal-expansion-voigt", alpha_final_voigt_sym,"1/K")
        self._add_key_to_current_property_instance("thermal-expansion-coefficient-names",unique_components_names)
        self._add_key_to_current_property_instance("thermal-expansion-coefficient-values",unique_components_values,"1/K")

        self._add_property_instance_and_common_crystal_genome_keys("volume-thermal-expansion-coefficient-crystal-npt",
                                                                   write_stress=True, write_temp=True)
        # Wallace, Thermodynamics of Crystals eq. 2.75
        self._add_key_to_current_property_instance(
            "volume-thermal-expansion-coefficient", alpha11 + alpha22 + alpha33, "1/K")