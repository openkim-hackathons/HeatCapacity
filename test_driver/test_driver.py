from concurrent.futures import as_completed, ProcessPoolExecutor
import os
import shutil
import subprocess
from typing import Optional, Tuple
from ase.io.lammpsdata import write_lammps_data
import numpy as np
from kim_tools import query_crystal_genome_structures
from kim_tools.test_driver import CrystalGenomeTestDriver
from helper_functions import (check_lammps_log_for_wrong_structure_format, compute_alpha, compute_heat_capacity,
                               get_cell_from_averaged_lammps_dump, get_positions_from_averaged_lammps_dump,
                               reduce_and_avg, run_lammps,get_position_hists)


class HeatCapacity(CrystalGenomeTestDriver):
    def _calculate(self, temperature: float, pressure: float, temperature_step_fraction: float,
                   number_symmetric_temperature_steps: int, timestep: float, number_sampling_timesteps: int,
                   repeat: Tuple[int, int, int] = (3, 3, 3), loose_triclinic_and_monoclinic=False,
                   max_workers: Optional[int] = None, **kwargs) -> None:
        """
        Compute constant-pressure heat capacity from centered finite difference (see Section 3.2 in
        https://pubs.acs.org/doi/10.1021/jp909762j).

        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.).
            This indicates which is being used for the current calculation.

        temperature:
            Temperature in Kelvin at which the heat capacity at constant pressure is estimated. Must be strictly greater
            than zero.

        pressure:
            Pressure in bar of the NPT simulation for the initial equilibration of the
            zero-temperature configuration. Must be strictly greater than zero.

        # TODO: Document arguments and add sensible default values.
        """
        # Check arguments.
        if not temperature > 0.0:
            raise RuntimeError("Temperature has to be larger than zero.")

        if not pressure > 0.0:
            raise RuntimeError("Pressure has to be larger than zero.")

        if not number_symmetric_temperature_steps > 0:
            raise RuntimeError("Number of symmetric temperature steps has to be bigger than zero.")

        if number_symmetric_temperature_steps * temperature_step_fraction >= 1.0:
            raise RuntimeError(
                "The given number of symmetric temperature steps and the given temperature-step fraction "
                "would yield zero or negative temperatures.")
        # TODO: Check all arguments.

        # Copy original atoms so that their information does not get lost when the new atoms are modified.
        atoms_new = self.atoms.copy()

        # UNCOMMENT THIS TO TEST A TRICLINIC STRUCTURE!
        # atoms_new = bulk('Ar', 'fcc', a=5.248)

        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_data
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        # Build supercell.
        atoms_new = atoms_new.repeat(repeat)

        # Get temperatures that should be simulated.
        temperature_step = temperature_step_fraction * temperature
        temperatures = [temperature + i * temperature_step
                        for i in range(-number_symmetric_temperature_steps, number_symmetric_temperature_steps + 1)]
        assert len(temperatures) == 2 * number_symmetric_temperature_steps + 1
        assert all(t > 0.0 for t in temperatures)

        # Write lammps file.
        TDdirectory = os.path.dirname(os.path.realpath(__file__))
        structure_file = os.path.join(TDdirectory, "output/zero_temperature_crystal.lmp")
        atoms_new.write(structure_file, format="lammps-data", masses=True)
        # Handle cases where kim models expect different structure file formats.
        try:
            run_lammps(self.kim_model_name, 0, temperatures[0], pressure, timestep,
                       number_sampling_timesteps, species, test_file_read=True)
        except subprocess.CalledProcessError as e:
            filename = "output/lammps_temperature_0.log"
            log_file = os.path.join(TDdirectory, filename)
            wrong_format_error = check_lammps_log_for_wrong_structure_format(log_file)

            if wrong_format_error:
                # write the atom configuration file in the in the 'charge' format some models expect
                write_lammps_data(structure_file, atoms_new, atom_style="charge", masses=True)
                # try to read the file again, raise any exeptions that might happen
                run_lammps(self.kim_model_name, 0, temperatures[0], pressure, timestep,
                           number_sampling_timesteps, species, test_file_read=True)

            else:
                raise e

        # Choose the correct accuracies file for kim-convergence based on whether the cell is orthogonal or not.
        if atoms_new.get_cell().orthorhombic:
            shutil.copyfile("accuracies_orthogonal.py", "accuracies.py")
        else:
            shutil.copyfile("accuracies_non_orthogonal.py", "accuracies.py")
        
        # Run Lammps simulations in parallel.
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, t in enumerate(temperatures):
                futures.append(executor.submit(
                    run_lammps, self.kim_model_name, i, t, pressure, timestep,
                    number_sampling_timesteps, species))
  
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
        for future, t in zip(futures, temperatures):
            assert future.done()
            assert future.exception() is None
            log_filename, restart_filename, average_position_filename, average_cell_filename = future.result()
            log_filenames.append(log_filename)
            restart_filenames.append(restart_filename)
            restart_filenames.append(restart_filename)
            atoms_new.set_cell(get_cell_from_averaged_lammps_dump(average_cell_filename))
            atoms_new.set_scaled_positions(
                get_positions_from_averaged_lammps_dump(average_position_filename))
            fstr = ".".join(os.path.basename(average_position_filename).split(".")[:-2])
            get_position_hists(atoms_new,repeat,fstr,mode = "reduce")
            get_position_hists(atoms_new,repeat,fstr,mode = "reduce_and_average")
            reduced_atoms = reduce_and_avg(atoms_new, repeat)
            
            crystal_genome_designation = self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(
                reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
    
        c = compute_heat_capacity(temperatures, log_filenames, 2)

        alpha = compute_alpha(log_filenames, temperatures, crystal_genome_designation["prototype_label"])
        # Print result.
        print('####################################')
        print('# NPT Heat Capacity Results #')
        print('####################################')
        print(f'C_p:\t{c}')
        print('####################################')
        print('# NPT Linear Thermal Expansion Tensor Results #')
        print('####################################')
        print(f'alpha:\t{alpha}')

        # TODO: We should write some coordinate file.
        self.poscar = None

        # Write property.
        self._add_property_instance_and_common_crystal_genome_keys(
            "heat-capacity-npt", write_stress=True, write_temp=True)  # last two default to False
        self._add_key_to_current_property_instance(
            "constant_pressure_heat_capacity", c["finite_difference_accuracy_2"][0], "eV/Kelvin",
            uncertainty_info={"source-std-uncert-value": c["finite_difference_accuracy_2"][1]})

        max_accuracy = len(temperatures) - 1
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

        self._add_property_instance_and_common_crystal_genome_keys("thermal-expansion-coefficient-npt",
                                                                   write_stress=True, write_temp=True)
        self._add_key_to_current_property_instance("alpha11", alpha11, "1/K", uncertainty_info={"source-std-uncert-value":alpha11_err})
        self._add_key_to_current_property_instance("alpha22", alpha22, "1/K", uncertainty_info={"source-std-uncert-value":alpha22_err})
        self._add_key_to_current_property_instance("alpha33", alpha33, "1/K", uncertainty_info={"source-std-uncert-value":alpha33_err})
        self._add_key_to_current_property_instance("alpha12", alpha12, "1/K", uncertainty_info={"source-std-uncert-value":alpha12_err})
        self._add_key_to_current_property_instance("alpha13", alpha13, "1/K", uncertainty_info={"source-std-uncert-value":alpha13_err})
        self._add_key_to_current_property_instance("alpha23", alpha23, "1/K", uncertainty_info={"source-std-uncert-value":alpha23_err})
        self._add_key_to_current_property_instance("thermal-expansion-coefficient", alpha_final, "1/K", uncertainty_info={"source-std-uncert-value":alpha_final_err})

        self.write_property_instances_to_file()
        print(c) 

if __name__ == "__main__":
    # model_name = "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    model_name = "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test_driver = HeatCapacity(model_name)
    list_of_queried_structures = query_crystal_genome_structures(kim_model_name=model_name,
                                                                 stoichiometric_species=['Al'],
                                                                 prototype_label='A_cF4_225_a')
    for queried_structure in list_of_queried_structures:
        test_driver(**queried_structure, temperature=293.15, pressure=1.0, temperature_step_fraction=0.01,
                    number_symmetric_temperature_steps=2, timestep=0.001, number_sampling_timesteps=100,
                    repeat=(3, 3, 3), loose_triclinic_and_monoclinic=True, max_workers=5)
