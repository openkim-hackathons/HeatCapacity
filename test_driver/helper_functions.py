import copy
from math import ceil, sqrt
import os
import re
import subprocess
from typing import Dict, Iterable, List, Tuple
from ase.cell import Cell
import findiff
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize


def run_lammps(modelname: str, temperature_index: int, temperature_K: float, pressure_bar: float, timestep_ps: float,
               number_sampling_timesteps: int, species: List[str],
               msd_threshold_angstrom_squared_per_hundred_timesteps: float, lammps_command: str,
               random_seed: int) -> Tuple[str, str, str, str]:
    """
    Run LAMMPS NPT simulation with the given parameters.

    After the simulation, this function plots the thermodynamic properties (volume, temperature, enthalpy).

    This function also processes the LAMMPS log file to extract equilibration information based on kim_convergence.
    It then computes the average atomic positions and cell parameters during the molecular-dynamics simulation, only
    considering data after equilibration.

    The given temperature_index is used to name the output files uniquely for each temperature in a temperature sweep.

    :param modelname:
        Name of the OpenKIM interatomic model.
    :type modelname: str
    :param temperature_index:
        Index of the temperature in the temperature sweep.
    :type temperature_index: int
    :param temperature_K:
        Target temperature in Kelvin.
    :type temperature_K: float
    :param pressure_bar:
        Target pressure in bars.
    :type pressure_bar: float
    :param timestep_ps:
        Timestep in picoseconds.
    :type timestep_ps: float
    :param number_sampling_timesteps:
        Number of timesteps for sampling thermodynamic quantities.
    :type number_sampling_timesteps: int
    :param species:
        List of chemical species in the system.
    :type species: List[str]
    :param msd_threshold_angstrom_squared_per_hundred_timesteps:
        Mean squared displacement threshold for vaporization in Angstroms^2 per 100*timestep.
    :type msd_threshold_angstrom_squared_per_hundred_timesteps: float
    :param lammps_command:
        Command to run LAMMPS (e.g., "mpirun -np 4 lmp_mpi" or "lmp").
    :type lammps_command: str
    :param random_seed:
        Random seed for velocity initialization.
    :type random_seed: int

    :return:
        A tuple containing paths to the LAMMPS log file, restart file, full average position file, and full average cell
        file.
    :rtype: Tuple[str, str, str, str]
    """
    pdamp = timestep_ps * 100.0
    tdamp = timestep_ps * 1000.0

    log_filename = f"output/lammps_temperature_{temperature_index}.log"
    restart_filename = f"output/final_configuration_temperature_{temperature_index}.restart"
    variables = {
        "modelname": modelname,
        "temperature": temperature_K,
        "temperature_seed": random_seed,
        "temperature_damping": tdamp,
        "pressure": pressure_bar,
        "pressure_damping": pdamp,
        "timestep": timestep_ps,
        "number_sampling_timesteps": number_sampling_timesteps,
        "species": " ".join(species),
        "average_position_filename": f"output/average_position_temperature_{temperature_index}.dump.*",
        "average_cell_filename": f"output/average_cell_temperature_{temperature_index}.dump",
        "write_restart_filename": restart_filename,
        "trajectory_filename": f"output/trajectory_{temperature_index}.lammpstrj",
        "msd_threshold": msd_threshold_angstrom_squared_per_hundred_timesteps
    }

    command = (
            f"{lammps_command} "
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items())
            + f" -log {log_filename}"
            + " -in npt.lammps")

    subprocess.run(command, check=True, shell=True)

    plot_property_from_lammps_log(log_filename, ("v_vol_metal", "v_temp_metal", "v_enthalpy_metal"))

    equilibration_time = extract_equilibration_step_from_logfile(log_filename)
    # Round to next multiple of 10000.
    equilibration_time = int(ceil(equilibration_time / 10000.0)) * 10000

    full_average_position_file = f"output/average_position_temperature_{temperature_index}.dump.full"
    compute_average_positions_from_lammps_dump("output",
                                               f"average_position_temperature_{temperature_index}.dump",
                                               full_average_position_file, equilibration_time)

    full_average_cell_file = f"output/average_cell_temperature_{temperature_index}.dump.full"
    compute_average_cell_from_lammps_dump(f"output/average_cell_temperature_{temperature_index}.dump",
                                          full_average_cell_file, equilibration_time)

    return log_filename, restart_filename, full_average_position_file, full_average_cell_file


def plot_property_from_lammps_log(in_file_path: str, property_names: Iterable[str]) -> None:
    """
    Extract and plot thermodynamic properties from the given Lammps log file.

    The extracted data is stored in a csv file with the same name as the log file but with a .csv extension.
    The plots of the specified properties against time are saved as property_name.png files.

    :param in_file_path:
        Path to the Lammps log file.
    :type in_file_path: str
    :param property_names:
        Iterable of thermodynamic property names to plot.
    :type property_names: Iterable[str]
    """
    def get_table(in_file):
        if not os.path.isfile(in_file):
            raise FileNotFoundError(in_file + " not found")
        elif ".log" not in in_file:
            raise FileNotFoundError("The file is not a *.log file")
        is_first_header = True
        header_flags = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
        eot_flags = ["Loop", "time", "on", "procs", "for", "steps"]
        table = []
        with open(in_file, "r") as f:
            line = f.readline()
            while line:  # Not EOF.
                is_header = True
                for _s in header_flags:
                    is_header = is_header and (_s in line)
                if is_header:
                    if is_first_header:
                        table.append(line)
                        is_first_header = False
                    content = f.readline()
                    while content:
                        is_eot = True
                        for _s in eot_flags:
                            is_eot = is_eot and (_s in content)
                        if not is_eot:
                            table.append(content)
                        else:
                            break
                        content = f.readline()
                line = f.readline()
        return table

    def write_table(table, out_file):
        with open(out_file, "w") as f:
            for l in table:
                f.writelines(l)

    dir_name = os.path.dirname(in_file_path)
    in_file_name = os.path.basename(in_file_path)
    out_file_path = os.path.join(dir_name, in_file_name.replace(".log", ".csv"))

    table = get_table(in_file_path)
    write_table(table, out_file_path)
    df = np.loadtxt(out_file_path, skiprows=1, usecols=tuple(range(16)))

    for property_name in property_names:
        with open(out_file_path) as file:
            first_line = file.readline().strip("\n")
        property_index = first_line.split().index(property_name)
        properties = df[:, property_index]
        step = df[:, 0]
        plt.plot(step, properties)
        plt.xlabel("step")
        plt.ylabel(property_name)
        img_file = os.path.join(dir_name, in_file_name.replace(".log", "_") + property_name + ".png")
        plt.savefig(img_file, bbox_inches="tight")
        plt.close()


def extract_equilibration_step_from_logfile(filename: str) -> int:
    """
    Extract the kim_convergence equilibration step from LAMMPS log file.

    :param filename:
        Path to the LAMMPS log file.
    :type filename: str

    :return:
        The equilibration step as an integer.
    :rtype: int
    """
    # Get file content.
    with open(filename, 'r') as file:
        data = file.read()

    # Look for pattern.
    exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
    mean_pattern = r'"equilibration_step"\s*([^ ]+)'
    match_init = re.search(exterior_pattern, data, re.DOTALL)
    equil_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
    if equil_matches is None:
        raise ValueError("Equilibration step not found")

    # Return largest match.
    return max(int(equil) for equil in equil_matches)


def compute_average_positions_from_lammps_dump(data_dir: str, file_str: str, output_filename: str,
                                               skip_steps: int) -> None:
    """
    Average atomic positions over multiple LAMMPS dump files.

    Within the given data directory, this function searches for dump files that start with the specified file string.
    After the filename, every dump file should end with a step number, e.g., average_position.dump.10000,
    average_position.dump.20000, etc. The function computes the average atomic positions across all these files,
    ignoring any files with step numbers less than or equal to the specified skip_steps. The resulting average
    positions are then written to the specified output file.

    :param data_dir:
        Directory containing the LAMMPS dump files.
    :type data_dir: str
    :param file_str:
        String that the dump files start with.
    :type file_str: str
    :param output_filename:
        Name of the output file to store the average positions.
    :type output_filename: str
    :param skip_steps:
        Step number threshold; dump files with steps less than or equal to this value are ignored.
    :type skip_steps: int
    """
    def get_id_pos_dict(file_name):
        id_pos_dict = {}
        header4N = ["NUMBER OF ATOMS"]
        header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
        is_table_started = False
        is_natom_read = False
        with open(file_name, "r") as f:
            line = f.readline()
            count_content_line = 0
            N = 0
            while line:
                if not is_natom_read:
                    is_natom_read = np.all([flag in line for flag in header4N])
                    if is_natom_read:
                        line = f.readline()
                        N = int(line)
                if not is_table_started:
                    contain_flags = np.all([flag in line for flag in header4pos])
                    is_table_started = contain_flags
                else:
                    count_content_line += 1
                    words = line.split()
                    id = int(words[0])
                    pos = np.array([float(words[1]), float(words[2]), float(words[3])])
                    id_pos_dict[id] = pos
                if count_content_line > 0 and count_content_line >= N:
                    break
                line = f.readline()
        if count_content_line < N:
            print("The file " + file_name +
                  " is not complete, the number of atoms is smaller than " + str(N))
        return id_pos_dict

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir + " does not exist")
    if not ".dump" in file_str:
        raise ValueError("file_str must be a string containing .dump")

    # Extract and store all the data.
    pos_list = []
    max_step, last_step_file = -1, ""
    for file_name in os.listdir(data_dir):
        if file_str in file_name:
            step = int(re.findall(r'\d+', file_name)[-1])
            if step <= skip_steps:
                continue
            file_path = os.path.join(data_dir, file_name)
            id_pos_dict = get_id_pos_dict(file_path)
            id_pos = sorted(id_pos_dict.items())
            id_list = [pair[0] for pair in id_pos]
            pos_list.append([pair[1] for pair in id_pos])
            # Check if this is the last step.
            if step > max_step:
                last_step_file, max_step = os.path.join(data_dir, file_name), step
    if max_step == -1 and last_step_file == "":
        raise RuntimeError("Found no files to average over.")
    pos_arr = np.array(pos_list)
    avg_pos = np.mean(pos_arr, axis=0)
    # Get the lines above the table from the file of the last step.
    with open(last_step_file, "r") as f:
        header4pos = ["id", "f_avePos[1]", "f_avePos[2]", "f_avePos[3]"]
        line = f.readline()
        description_str = ""
        is_table_started = False
        while line:
            description_str += line
            is_table_started = np.all([flag in line for flag in header4pos])
            if is_table_started:
                break
            else:
                line = f.readline()
    # Write the output to the file.
    with open(output_filename, "w") as f:
        f.write(description_str)
        for i in range(len(id_list)):
            f.write(str(id_list[i]))
            f.write("  ")
            for dim in range(3):
                f.write('{:3.6}'.format(avg_pos[i, dim]))
                f.write("  ")
            f.write("\n")


def compute_average_cell_from_lammps_dump(input_file: str, output_file: str, skip_steps: int) -> None:
    """
    Average the cell from the given input file.

    This function computes the average cell across a LAMMPS dump file containing the cell information over time,
    ignoring any cell information at step numbers less than or equal to the specified skip_steps. The resulting average
    cell is then written to the specified output file.

    :param input_file:
        Path to the LAMMPS dump file containing cell information.
    :type input_file: str
    :param output_file:
        Name of the output file to store the average cell.
    :type output_file: str
    :param skip_steps:
        Step number threshold; dump files with steps less than or equal to this value are ignored.
    :type skip_steps: int
    """
    with open(input_file, "r") as f:
        f.readline()  # Skip the first line.
        header = f.readline()
        header = header.replace("#", "")
    property_names = header.split()
    data = np.loadtxt(input_file, skiprows=2)
    time_step_index = property_names.index("TimeStep")
    time_step_data = data[:, time_step_index]
    cutoff_index = np.argmax(time_step_data > skip_steps)
    assert time_step_data[cutoff_index] > skip_steps
    assert cutoff_index == 0 or time_step_data[cutoff_index - 1] <= skip_steps
    mean_data = data[cutoff_index:].mean(axis=0).tolist()
    with open(output_file, "w") as f:
        print("# Full time-averaged data for cell information", file=f)
        print(f"# {' '.join(name for name in property_names if name != 'TimeStep')}", file=f)
        print(" ".join(str(mean_data[i]) for i, name in enumerate(property_names) if name != "TimeStep"), file=f)


def get_positions_from_averaged_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
    """
    Helper function to extract positions from the averaged LAMMPS dump file.

    :param filename:
        Path to the averaged LAMMPS dump file.
    :type filename: str

    :return:
        A list of tuples representing the (x, y, z) positions of atoms.
    :rtype: List[Tuple[float, float, float]]
    """
    lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
    return [(line[1], line[2], line[3]) for line in lines]


def get_cell_from_averaged_lammps_dump(filename: str) -> npt.NDArray[np.float64]:
    """
    Helper function to extract the cell from the averaged LAMMPS dump file.

    :param filename:
        Path to the averaged LAMMPS dump file.
    :type filename: str

    :return:
        A 3x3 numpy array representing the cell vectors.
    :rtype: npt.NDArray[np.float64]
    """
    cell_list = np.loadtxt(filename, comments='#')
    assert len(cell_list) == 6
    cell = np.empty(shape=(3, 3))
    cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
    cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
    cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
    return cell


def compute_heat_capacity(temperatures: List[float], log_filenames: List[str],
                          enthalpy_index: int) -> Dict[str, Tuple[float, float]]:
    """
    Compute the heat capacity and its error from LAMMPS log files at different temperatures.

    This function assumes the kim-convergence was used to make sure that the enthalpy data is equilibrated.
    Kim-convergence will then print the mean and 95% confidence interval of the enthalpy to the log file.
    The quantity_index specifies the index of the enthalpy in kim-convergence.

    This function uses two methods to estimate the heat capacity:
    1. Finite differences with varying accuracy (2, 4, ..., up to the maximum possible accuracy based on the number of
       temperatures).
    2. Linear fit to the enthalpy vs. temperature data.

    :param temperatures:
        List of temperatures corresponding to the log files.
    :type temperatures: List[float]
    :param log_filenames:
        List of LAMMPS log file paths.
    :type log_filenames: List[str]
    :param enthalpy_index:
        Index of the enthalpy quantity in the kim-convergence output.
    :type enthalpy_index: int

    :return:
        A dictionary where keys are method names (e.g., "finite_difference_accuracy_2", "fit") and values are tuples
        containing the estimated heat capacity and its error.
    :rtype: Dict[str, Tuple[float, float]]
    """
    enthalpy_means = []
    enthalpy_errs = []
    for log_filename in log_filenames:
        enthalpy_mean, enthalpy_conf = extract_mean_error_from_logfile(log_filename, enthalpy_index)
        enthalpy_means.append(enthalpy_mean)
        # Correct 95% confidence interval to standard error.
        enthalpy_errs.append(enthalpy_conf / 1.96)

    # Use finite differences to estimate derivative.
    temperature_step = temperatures[1] - temperatures[0]
    assert all(abs(temperatures[i + 1] - temperatures[i] - temperature_step)
               < 1.0e-12 for i in range(len(temperatures) - 1))
    assert len(temperatures) >= 3
    max_accuracy = len(temperatures) - 1
    heat_capacity = {}
    for accuracy in range(2, max_accuracy + 1, 2):
        heat_capacity[
            f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(
            temperature_step, enthalpy_means, enthalpy_errs, accuracy)

    # Use linear fit to estimate derivative.
    heat_capacity["fit"] = get_slope_and_error(
        temperatures, enthalpy_means, enthalpy_errs)

    return heat_capacity


def extract_mean_error_from_logfile(filename: str, quantity_index: int) -> Tuple[float, float]:
    """
    Extract the mean and error (95% confidence interval) of a quantity from LAMMPS log file.

    This function assumes the kim-convergence was used to make sure that the quantity data is equilibrated.
    Kim-convergence will then print the mean and 95% confidence interval of the quantity to the log file.
    The quantity_index specifies the index of the quantity in kim-convergence.

    :param filename:
        Path to the LAMMPS log file.
    :type filename: str
    :param quantity_index:
        Index of the quantity in the kim-convergence output.
    :type quantity_index: int

    :return:
        A tuple containing the mean and error of the quantity.
    :rtype: Tuple[float, float]
    """
    # Get content.
    with open(filename, "r") as file:
        data = file.read()

    # Look for print pattern.
    exterior_pattern = r'print "\${run_var}"\s*\{(.*?)\}\s*variable run_var delete'
    mean_pattern = r'"mean"\s*([^ ]+)'
    error_pattern = r'"upper_confidence_limit"\s*([^ ]+)'
    match_init = re.search(exterior_pattern, data, re.DOTALL)
    mean_matches = re.findall(mean_pattern, match_init.group(), re.DOTALL)
    error_matches = re.findall(error_pattern, match_init.group(), re.DOTALL)
    if mean_matches is None:
        raise ValueError("Mean not found")
    if error_matches is None:
        raise ValueError("Error not found")

    # Get correct match.
    mean = float(mean_matches[quantity_index])
    error = float(error_matches[quantity_index])

    return mean, error


def get_slope_and_error(x_values: List[float], y_values: List[float], y_errs: List[float]):
    """
    Fit a line to the given data and return the slope and its error.

    :param x_values:
        List of x values.
    :type x_values: List[float]
    :param y_values:
        List of y values.
    :type y_values: List[float]
    :param y_errs:
        List of y errors.
    :type y_errs: List[float]

    :return:
        A tuple containing the slope and its error.
    :rtype: Tuple[float, float]
    """
    # noinspection PyUnresolvedReferences
    popt, pcov = scipy.optimize.curve_fit(lambda x, m, b: m * x + b, x_values, y_values,
                                          sigma=y_errs, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def get_center_finite_difference_and_error(diff_x: float, y_values: List[float], y_errs: List[float],
                                           accuracy: int) -> Tuple[float, float]:
    """
    Estimate the derivative at the center of the given series of data points with finite differences and propagate the
    error.

    This function uses the findiff package to get the finite difference coefficients for the specified accuracy.

    :param diff_x:
        The difference in the x values.
    :type diff_x: float
    :param y_values:
        List of y values.
    :type y_values: List[float]
    :param y_errs:
        List of y errors.
    :type y_errs: List[float]
    :param accuracy:
        The desired accuracy of the finite difference.
    :type accuracy: int

    :return:
        A tuple containing the finite difference and its error.
    :rtype: Tuple[float, float]
    """
    assert len(y_values) == len(y_errs)
    assert len(y_values) > accuracy
    assert len(y_values) % 2 == 1
    center_index = len(y_values) // 2
    coefficients = findiff.coefficients(deriv=1, acc=accuracy)["center"]["coefficients"]
    offsets = findiff.coefficients(deriv=1, acc=accuracy)["center"]["offsets"]
    finite_difference = 0.0
    finite_difference_error_squared = 0.0
    for coefficient, offset in zip(coefficients, offsets):
        finite_difference += coefficient * y_values[center_index + offset]
        finite_difference_error_squared += (coefficient * y_errs[center_index + offset]) ** 2
    finite_difference /= diff_x
    finite_difference_error_squared /= (diff_x * diff_x)
    return finite_difference, sqrt(finite_difference_error_squared)


def compute_alpha_tensor(new_cells: list[Cell], temperatures:list[float]) -> np.ndarray:
    """
    Compute the thermal expansion tensor of a crystal given an array of unit cells
    equilibrated at uniformly spaced temperatures.

    Compute the deformation matrix between the center cell and other cells, 
    and from that compute the strain matrix. The thermal expansion tensor is computed 
    by computing the gradient of the strain as a function temperature.

    :param new_cells: An array of Cell objects equilibrated at uniformly spaced temperatures, 
        centered on a target temperature.
    :type new_cells:list[ase.cell.Cell]
    :param temperatures: List of temperatures that the cells were equilibrated at
    :type temperatures: list[float]
    """
    dim = 3

    temperature_step = temperatures[1] - temperatures[0]
    assert all(abs(temperatures[i + 1] - temperatures[i] - temperature_step)
               < 1.0e-12 for i in range(len(temperatures) - 1))
    assert len(temperatures) >= 3
    max_accuracy = len(temperatures) - 1

    center_cell = new_cells[int(np.floor(len(new_cells)/2))]

    center_cell_inverse = np.linalg.inv(center_cell)

    strains=[]

    # calculate the strain matrix
    for index in range(len(temperatures)):

        new_cell = new_cells[index]

        # calculate the deformation matrix from the old and new cells
        deformation = np.matmul(new_cell, center_cell_inverse) - np.identity(dim)

        strain = np.empty((dim,dim))

        for i in range(dim):
            for j in range(dim):
                
                sum_term=0
                for k in range(dim):
                    sum_term += deformation[k,i]*deformation[k,j]

                strain[i,j]=0.5*(deformation[i,j]+deformation[j,i]+sum_term)
        
        strains.append(strain)

    zero = {}
    for accuracy in range(2, max_accuracy + 1, 2):
        zero[f"finite_difference_accuracy_{accuracy}"] = [0.0, 0.0]

    alpha11 = copy.deepcopy(zero)
    alpha22 = copy.deepcopy(zero)
    alpha33 = copy.deepcopy(zero)
    alpha23 = copy.deepcopy(zero)
    alpha13 = copy.deepcopy(zero)
    alpha12 = copy.deepcopy(zero)


    for accuracy in range(2, max_accuracy + 1, 2):

        strain11_temps=[]
        strain22_temps=[]
        strain33_temps=[]
        strain23_temps=[]
        strain13_temps=[]
        strain12_temps=[]

        
        for t in range(len(temperatures)):

            strain11=strains[t][0,0]
            strain22=strains[t][1,1]
            strain33=strains[t][2,2]
            strain23=strains[t][1,2]
            strain13=strains[t][0,2]
            strain12=strains[t][0,1]

            strain11_temps.append(strain11)
            strain22_temps.append(strain22)
            strain33_temps.append(strain33)
            strain23_temps.append(strain23)
            strain13_temps.append(strain13)
            strain12_temps.append(strain12)

        # TODO: figure out how to calculate uncertianties
        strain_errs = list(np.zeros(len(strain11_temps)))

        alpha11[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain11_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)
        alpha22[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain22_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)
        alpha33[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain33_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)
        alpha23[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain23_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)
        alpha13[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain13_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)
        alpha12[f"finite_difference_accuracy_{accuracy}"] = get_center_finite_difference_and_error(temperature_step,
                                                                                                   strain12_temps,
                                                                                                   strain_errs,
                                                                                                   accuracy)


    # enforce tensor symmetries
    alpha21 = alpha12
    alpha31 = alpha13
    alpha32 = alpha23

    alpha = np.array([[alpha11, alpha12, alpha13],
                      [alpha21, alpha22, alpha23],
                      [alpha31, alpha32, alpha33]])

    # thermal expansion coeff tensor
    return alpha
