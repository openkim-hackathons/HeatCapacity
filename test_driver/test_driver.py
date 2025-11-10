from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import random
import shutil
from typing import Optional, Sequence
from ase.calculators.lammps import convert, Prism
import numpy as np
from kim_tools import get_stoich_reduced_list_from_prototype, KIMTestDriverError
from kim_tools.symmetry_util.core import (reduce_and_avg, kstest_reduced_distances, 
                                          PeriodExtensionException, fit_voigt_tensor_to_cell_and_space_group_symb, full_to_voigt_symb)
from kim_tools.test_driver import SingleCrystalTestDriver
from .helper_functions import (compute_alpha_tensor, compute_heat_capacity, get_cell_from_averaged_lammps_dump,
                               get_positions_from_averaged_lammps_dump, run_lammps)


class TestDriver(SingleCrystalTestDriver):
    def _calculate(self, temperature_step_fraction: float, number_symmetric_temperature_steps: int, timestep: float,
                   number_sampling_timesteps: int = 100, repeat: Sequence[int] = (3, 3, 3),
                   max_workers: Optional[int] = None, lammps_command = "lmp", msd_threshold: float = 0.1,
                   random_seeds: Optional[Sequence[int]] = None, **kwargs) -> None:
        """
        Compute constant-pressure heat capacity from centered finite difference (see Section 3.2 in
        https://pubs.acs.org/doi/10.1021/jp909762j).
        """
        # Set prototype label
        self.prototype_label = self._get_nominal_crystal_structure_npt()["prototype-label"]["source-value"]

        # Get temperature in Kelvin.
        temperature_K = self._get_temperature(unit="K")

        # Get cauchy stress tensor in bar.
        cell_cauchy_stress_bar = self._get_cell_cauchy_stress(unit="bar")

        # Check arguments.
        if not temperature_K > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")

        if not len(cell_cauchy_stress_bar) == 6:
            raise RuntimeError("Specify all six (x, y, z, xy, xz, yz) entries of the cauchy stress tensor.")

        if not (cell_cauchy_stress_bar[0] == cell_cauchy_stress_bar[1] == cell_cauchy_stress_bar[2]):
            raise RuntimeError("The diagonal entries of the stress tensor have to be equal so that a hydrostatic "
                               "pressure is used.")

        if not (cell_cauchy_stress_bar[3] == cell_cauchy_stress_bar[4] == cell_cauchy_stress_bar[5] == 0.0):
            raise RuntimeError("The off-diagonal entries of the stress tensor have to be zero so that a hydrostatic "
                               "pressure is used.")

        if not number_symmetric_temperature_steps > 0:
            raise RuntimeError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise RuntimeError(
                "The given number of symmetric temperature steps and the given temperature-step fraction "
                "would yield zero or negative temperatures.")

        if not number_sampling_timesteps > 0:
            raise RuntimeError("Number of timesteps between sampling in Lammps has to be bigger than zero.")

        if not len(repeat) == 3:
            raise RuntimeError("The repeat argument has to be a tuple of three integers.")

        if not all(r >= 0 for r in repeat):
            raise RuntimeError("All number of repeats must be bigger than zero.")

        if max_workers is not None and not max_workers > 0:
            raise RuntimeError("Maximum number of workers has to be bigger than zero.")
        
        if not msd_threshold > 0.0:
            raise RuntimeError("The mean-squared displacement threshold has to be bigger than zero.")

        if random_seeds is not None:
            if len(random_seeds) != 2 * number_symmetric_temperature_steps + 1:
                raise RuntimeError("If random seeds are given, their number has to match the number of temperatures "
                                   "being simulated.")
            if not all(rs > 0 for rs in random_seeds):
                raise RuntimeError("All random seeds must be bigger than zero.")
        else:
            # Get random 31-bit unsigned integers.
            random_seeds = [random.getrandbits(31) for _ in range(2 * number_symmetric_temperature_steps + 1)]

        # Get pressure from cauchy stress tensor.
        pressure_bar = -cell_cauchy_stress_bar[0]

         # Copy original atoms so that their information does not get lost.
        original_atoms = self._get_atoms()

        # Create atoms object that will contain the supercell.
        atoms_new = self._get_atoms()

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        # Build supercell.
        if repeat == (0, 0, 0):
            # Get a size close to 10K atoms (shown to give good convergence)
            x = int(np.ceil(np.cbrt(10000 / len(atoms_new))))
            repeat = (x, x, x)

        atoms_new = atoms_new.repeat(repeat)
        
        # Determine appropriate number of processors based on system size.
        number_atoms = len(atoms_new)

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * temperature_K
        temperatures = [temperature_K + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

        # Create output directory for all data files and copy over necessary files.
        os.makedirs("output", exist_ok=True)
        test_driver_directory = os.path.dirname(os.path.realpath(__file__))
        if os.getcwd() != test_driver_directory:
            shutil.copyfile(os.path.join(test_driver_directory, "npt.lammps"), "npt.lammps")
            shutil.copyfile(os.path.join(test_driver_directory, "run_length_control.py"), "run_length_control.py")
        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        with open("accuracies.py", "w") as file:
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

        # Write lammps file.
        structure_file = "output/zero_temperature_crystal.lmp"
        atom_style = self._get_supported_lammps_atom_style()
        atoms_new.write(structure_file, format="lammps-data", masses=True, units="metal", atom_style=atom_style)

        # Run Lammps simulations in parallel.
        assert len(temperatures) == len(random_seeds)
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, (t, rs) in enumerate(zip(temperatures, random_seeds)):
                futures.append(executor.submit(
                    run_lammps, self.kim_model_name, i, t, pressure_bar, timestep,
                    number_sampling_timesteps, species, msd_threshold, lammps_command, rs))

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
        restart_filenames = []
        all_cells = []
        middle_temperature_atoms = None
        middle_temperature = None
        for t_index, (future, t) in enumerate(zip(futures, temperatures)):
            assert future.done()
            assert future.exception() is None
            log_filename, restart_filename, average_position_filename, average_cell_filename = future.result()
            log_filenames.append(log_filename)
            restart_filenames.append(restart_filename)
            restart_filenames.append(restart_filename)
            with open(log_filename, "r") as f:
                for line in f:
                    if line.startswith("Crystal melted or vaporized"):
                        raise KIMTestDriverError(f"Crystal melted or vaporized during simulation at temperature {t} K.")
            atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
            atoms_new.set_scaled_positions(
                get_positions_from_averaged_lammps_dump(average_position_filename))
            reduced_atoms, reduced_distances = reduce_and_avg(atoms_new, repeat)

            if t_index == number_symmetric_temperature_steps:
                # Store the atoms of the middle temperature for later because their crystal genome designation 
                # will be used for the heat-capacity and thermal expansion properties.
                middle_temperature_atoms = reduced_atoms.copy()
                middle_temperature = t
            
            # Check that the symmetry of the structure did not change.
            if not self._verify_unchanged_symmetry(reduced_atoms):
                reduced_atoms.write(f"output/reduced_atoms_temperature_{t_index}_failing.poscar",
                                    format="vasp", sort=True)
                raise KIMTestDriverError(f"Symmetry of structure changed during simulation at temperature {t} K.")
            # Check that the reduced distances are normally distributed.
            try:
                kstest_reduced_distances(reduced_distances, significance_level=0.05,
                                         plot_filename=f"output/reduced_distance_histogram_temperature_{t_index}.pdf",
                                         number_bins=20)
            except PeriodExtensionException as e:
                reduced_atoms.write(f"output/reduced_atoms_temperature_{t_index}_failing.poscar",
                                    format="vasp", sort=True)
                raise KIMTestDriverError(f"Reduced distances are not normally distributed at temperature {t} K: {e}")
            
            # Write NPT crystal structures.
            self._update_nominal_parameter_values(reduced_atoms)
            all_cells.append(self._get_atoms().cell)
            self._add_property_instance_and_common_crystal_genome_keys("crystal-structure-npt", write_stress=True,
                                                                       write_temp=t)
            self._add_file_to_current_property_instance("restart-file",
                                                        f"output/final_configuration_temperature_{t_index}.restart")
            
            # Reset to original atoms.
            self._update_nominal_parameter_values(original_atoms)

        assert middle_temperature_atoms is not None
        assert middle_temperature is not None

        c = compute_heat_capacity(temperatures, log_filenames, 2)
        alpha = compute_alpha_tensor(all_cells,temperatures)

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
        self._update_nominal_parameter_values(middle_temperature_atoms)
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-npt", write_stress=True, write_temp=middle_temperature)
        assert len(atoms_new) == len(original_atoms) * repeat[0] * repeat[1] * repeat[2]
        number_atoms = len(atoms_new)
        self._add_key_to_current_property_instance(
            "constant-pressure-heat-capacity-per-atom",
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / number_atoms,
            "eV/Kelvin",
            uncertainty_info={
                "source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / number_atoms})
        number_atoms_in_formula = sum(get_stoich_reduced_list_from_prototype(self.prototype_label))
        assert number_atoms % number_atoms_in_formula == 0
        number_formula = number_atoms // number_atoms_in_formula
        self._add_key_to_current_property_instance(
            "constant-pressure-heat-capacity-per-formula",
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / number_formula,
            "eV/Kelvin",
            uncertainty_info={
                "source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / number_formula})
        total_mass_g_per_mol = sum(atoms_new.get_masses())
        self._add_key_to_current_property_instance(
            "constant-pressure-specific-heat-capacity",
            c[f"finite_difference_accuracy_{max_accuracy}"][0] / total_mass_g_per_mol,
            "eV/Kelvin/(g/mol)",
            uncertainty_info={
                "source-std-uncert-value": c[f"finite_difference_accuracy_{max_accuracy}"][1] / total_mass_g_per_mol})

        alpha11 = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha11_err = alpha[0][0][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha12 = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha12_err = alpha[0][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha13 = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha13_err = alpha[0][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha22 = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha22_err = alpha[1][1][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha23 = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha23_err = alpha[1][2][f"finite_difference_accuracy_{max_accuracy}"][1]
        alpha33 = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][0]
        alpha33_err = alpha[2][2][f"finite_difference_accuracy_{max_accuracy}"][1]

        # enforce tensor symmetries
        alpha21 = alpha12
        alpha31 = alpha13
        alpha32 = alpha23

        alpha21_err = alpha12_err
        alpha31_err = alpha13_err
        alpha32_err = alpha23_err

        alpha_final = np.asarray([[alpha11, alpha12, alpha13],
                                  [alpha21, alpha22, alpha23],
                                  [alpha31, alpha32, alpha33]])

        alpha_final_err = np.asarray([[alpha11_err, alpha12_err, alpha13_err],
                                      [alpha21_err, alpha22_err, alpha23_err],
                                      [alpha31_err, alpha32_err, alpha33_err]])

        self._add_property_instance_and_common_crystal_genome_keys("tag:staff@noreply.openkim.org,2024-03-11:property/thermal-expansion-coefficient-npt",
                                                                   write_stress=True, write_temp=True)
        space_group = int(self.prototype_label.split("_")[2])

        # thermal expansion tensor in voigt notation
        alpha_final_voigt = full_to_voigt_symb(alpha_final)

        # TODO: upgrade to fit_voigt_tensor_and_error_to_cell_and_space_group()
        # once errors are being calculated
        alpha_final_voigt_sym = fit_voigt_tensor_to_cell_and_space_group_symb(alpha_final_voigt,
                                                                               middle_temperature_atoms.get_cell(),
                                                                               space_group)
        
        alpha_final_voigt_sym = alpha_final_voigt
        # alpha11 unique for all space groups
        unique_components_names = ["alpha11"]
        unique_components_values = [alpha11]
        unique_components_errs = [alpha11_err]

        # hexagona, trigonal, tetragonal space groups alpha33 also unique
        if space_group <= 194:

            unique_components_names.append("alpha33")
            unique_components_values.append(alpha33)
            unique_components_errs.append(alpha33_err)

        # orthorhombic, alpha22 also unique
        if space_group <= 74:

            # insert alpha22 in the middle so they end up sorted
            # into voigt notation order
            unique_components_names.insert(1,"alpha22")
            unique_components_values.insert(1,alpha22)
            unique_components_errs.insert(1,alpha22_err)

        # monoclinic or triclinic, all components potentially unique
        if space_group <= 15:

            unique_components_names.append("alpha23")
            unique_components_names.append("alpha13")
            unique_components_names.append("alpha12")

            unique_components_values.append(alpha23)
            unique_components_values.append(alpha13)
            unique_components_values.append(alpha12)

            unique_components_errs.append(alpha23_err)
            unique_components_errs.append(alpha13_err)
            unique_components_errs.append(alpha12_err)

        # TODO: add uncertainty info once we decide how to calculate cell errors
        self._add_key_to_current_property_instance("thermal-expansion-tensor-voigt", alpha_final_voigt, "1/K")
        self._add_key_to_current_property_instance("thermal-expansion-tensor-voigt-symmetry-reduced", alpha_final_voigt_sym,"1/K")
        self._add_key_to_current_property_instance("unique-components-names",unique_components_names)
        self._add_key_to_current_property_instance("unique-components-values",unique_components_values,"1/K")
