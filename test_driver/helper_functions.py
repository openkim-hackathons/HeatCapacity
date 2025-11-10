import copy
from math import ceil, sqrt
import os
import random
import re
import subprocess
from typing import Dict, Iterable, List, Tuple
from ase import Atoms
import findiff
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize


def run_lammps(modelname: str, temperature_index: int, temperature: float, pressure: float, timestep: float,
               number_sampling_timesteps: int, species: List[str], msd_threshold: float,
               lammps_command: str) -> Tuple[str, str, str, str]:
    # Get random 31-bit unsigned integer.
    seed = random.getrandbits(31)

    pdamp = timestep * 100.0
    tdamp = timestep * 1000.0

    log_filename = f"output/lammps_temperature_{temperature_index}.log"
    restart_filename = f"output/final_configuration_temperature_{temperature_index}.restart"
    variables = {
        "modelname": modelname,
        "temperature": temperature,
        "temperature_seed": seed,
        "temperature_damping": tdamp,
        "pressure": pressure,
        "pressure_damping": pdamp,
        "timestep": timestep,
        "number_sampling_timesteps": number_sampling_timesteps,
        "species": " ".join(species),
        "average_position_filename": f"output/average_position_temperature_{temperature_index}.dump.*",
        "average_cell_filename": f"output/average_cell_temperature_{temperature_index}.dump",
        "write_restart_filename": restart_filename,
        "trajectory_filename": f"output/trajectory_{temperature_index}.lammpstrj",
        "msd_threshold": msd_threshold
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
    The function to get the value of the property with time from ***.log
    the extracted data are stored as ***.csv and ploted as property_name.png
    data_dir --- the directory contains lammps_equilibration.log
    property_names --- the list of properties
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
    This function compute the average position over *.dump files which contains the file_str in data_dir and output it
    to data_dir/[file_str]_over_dump.out

    input:
    data_dir -- the directory contains all the data e.g average_position.dump.* files
    file_str -- the files whose names contain the file_str are considered
    output_filename -- the name of the output file
    skip_steps -- dump files with steps <= skip_steps are ignored
    """

    def get_id_pos_dict(file_name):
        '''
        input:
        file_name--the file_name that contains average postion data
        output:
        the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
        for the averaged positions over files
        '''
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
    lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key=lambda x: x[0])
    return [(line[1], line[2], line[3]) for line in lines]


def get_cell_from_averaged_lammps_dump(filename: str) -> npt.NDArray[np.float64]:
    cell_list = np.loadtxt(filename, comments='#')
    assert len(cell_list) == 6
    cell = np.empty(shape=(3, 3))
    cell[0, :] = np.array([cell_list[0], 0.0, 0.0])
    cell[1, :] = np.array([cell_list[3], cell_list[1], 0.0])
    cell[2, :] = np.array([cell_list[4], cell_list[5], cell_list[2]])
    return cell


def compute_heat_capacity(temperatures: List[float], log_filenames: List[str],
                          quantity_index: int) -> Dict[str, Tuple[float, float]]:
    enthalpy_means = []
    enthalpy_errs = []
    for log_filename in log_filenames:
        enthalpy_mean, enthalpy_conf = extract_mean_error_from_logfile(log_filename, quantity_index)
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


def extract_mean_error_from_logfile(filename: str, quantity: int) -> Tuple[float, float]:
    """
    Function to extract the average from a LAAMPS log file for a given quantity

    @param filename : name of file
    @param quantity : quantity to take from
    @return mean : reported mean value
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
    mean = float(mean_matches[quantity])
    error = float(error_matches[quantity])

    return mean, error


def get_slope_and_error(x_values: List[float], y_values: List[float], y_errs: List[float]):
    popt, pcov = scipy.optimize.curve_fit(lambda x, m, b: m * x + b, x_values, y_values,
                                          sigma=y_errs, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def get_center_finite_difference_and_error(diff_x: float, y_values: List[float], y_errs: List[float],
                                           accuracy: int) -> Tuple[float, float]:
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

def compute_alpha_tensor(new_cells: list[Atoms.cell],
                         temperatures:list[float]):
    
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
        strain_errs=np.zeros(len(strain11_temps))

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

def check_lammps_log_for_wrong_structure_format(log_file):
    wrong_format_in_structure_file = False

    try:
        with open(log_file, "r") as logfile:
            data = logfile.read()
            data = data.split("\n")
            final_line = data[-2]

            if final_line == "Last command: read_data output/zero_temperature_crystal.lmp":
                wrong_format_in_structure_file = True
    except FileNotFoundError:
        pass

    return wrong_format_in_structure_file
