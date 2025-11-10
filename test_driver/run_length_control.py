"""Run length control for LAMMPS."""
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from typing import Optional
from lammps import lammps
import kim_convergence as cr
from accuracies import RELATIVE_ACCURACY, ABSOLUTE_ACCURACY

# Initial run length
INITIAL_RUN_LENGTH: int = 10000
# Run length increasing factor
RUN_LENGTH_FACTOR: float = 1
# The maximum run length represents a cost constraint.
MAX_RUN_LENGTH: int = 1000 * INITIAL_RUN_LENGTH
# The maximum number of steps as an equilibration hard limit. If the
# algorithm finds equilibration_step greater than this limit it will fail.
# For the default None, the function is using `maximum_run_length // 2` as
# the maximum equilibration step.
MAX_EQUILIBRATION_STEP: Optional[int] = 500 * INITIAL_RUN_LENGTH
# Maximum number of independent samples.
MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES: Optional[int] = 1000
# Probability (or confidence interval) and must be between 0.0 and 1.0, and
# represents the confidence for calculation of relative halfwidths estimation.
CONFIDENCE: float = 0.95
# Method to use for approximating the upper confidence limit of the mean.
UCL_METHOD: str = 'uncorrelated_sample'
# if ``True``, dump the final trajectory data to a file.
DUMP_TRAJECTORY: bool = False


# Do not modify
_LAMMPS_ARGUMENTS = (
    'variable',
    'compute',
    'fix',
    'lb',
    'lbound',
    'ub',
    'ubound',
    'mean',
    'population_mean',
    'std',
    'population_std',
    'cdf',
    'population_cdf',
    'args',
    'population_args',
    'loc',
    'population_loc',
    'scale',
    'population_scale',
)

# Do not modify
_PREFIX_NAME = {
    'v_': 'variable',
    'c_': 'computation',
    'f_': 'fix',
}


def run_length_control(lmpptr, nevery: int, *argv) -> None:
    """Control the length of the LAMMPS simulation run.

    Arguments:
        lmpptr {pointer} -- LAMMPS pointer to a previously created LAMMPS
            object.
        nevery {int} -- use input values every this many timesteps. It
            specifies on what timesteps the input values will be used in
            order to be stored. Only timesteps that are a multiple of nevery,
            including timestep 0, will contribute values.

    Note:
        Each input value throug argv can be the result of a `compute` or
        a `fix` or the evaluation of an equal-style or vector-style `variable`.
        In each case, the `compute`, `fix`, or `variable` must produce a
        global quantity, not a per-atom or local quantity. And the global
        quantity must be a scalar, not a vector or array.

        ``Computes`` that produce global quantities are those which do not have
        the word atom in their style name. Only a few fixes produce global
        quantities.

        ``Variables of style equal or vector`` are the only ones that can be
        used as an input here. ``Variables of style atom`` cannot be used,
        since they produce per-atom values.

        Each input value through argv following the argument `lb`, or `lbound`
        and `ub`, or `ubound` must previously be defined in the input script
        as the evaluation of an equal-style `variable`.

    """
    lmp = lammps(ptr=lmpptr)

    cr.cr_check(nevery, 'nevery', int, 1)

    cmd = 'fix cr_fix all vector {} '.format(nevery)

    # Arguments
    arguments_map = {}

    # New keyword
    ctrl_map = {}

    # default prefix
    prefix = 'v_'

    var_name = None

    # population info
    population_mean = None
    population_std = None
    population_cdf = None
    population_args = None
    population_loc = None
    population_scale = None

    number_of_arguments = len(argv)

    argument_counter = 0
    i = 0
    while i < number_of_arguments:
        arg = argv[i]

        if arg not in _LAMMPS_ARGUMENTS:
            msg = 'The input argument "{}" is not suppoerted.'
            raise cr.CRError(msg)

        if arg in ('variable', 'compute', 'fix'):
            # The value following the argument `variable`, `compute`, or
            # `fix` must previously (in the input script) be defined
            # (prefixed) as `v_`, `c_`, or `f_` variable respectively.
            prefix = arg[0] + '_'

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = {
                    'v_': 'name of variable',
                    'c_': 'user-assigned name for the computation',
                    'f_': 'user-assigned name for the fix',
                }
                raise cr.CRError(msg[prefix] + ' is not provided.')

            var_name = '{}{}'.format(prefix, arg)

            arguments_map[argument_counter] = var_name
            argument_counter += 1

            cmd = cmd + var_name + ' '

            i += 1
            continue

        if var_name is None:
            msg = 'A `variable` or a `compute`, or a `fix` must '
            msg += 'previously be defined.'
            raise cr.CRError(msg)

        if arg in ('lbound', 'lb'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'lower bound is not provided.'
                raise cr.CRError(msg)

            # parse the value
            try:
                var_lb = float(arg)
            except ValueError:
                # lb can be an equal-style variable
                var_lb = lmp.extract_variable(arg, None, 0)

                if var_lb is None:
                    msg = 'lb must be followed by an equal-style variable.'
                    raise cr.CRError(msg)

            var_ub = None

            i += 1
            try:
                arg = argv[i]
            except IndexError:
                ctrl_map[var_name] = tuple([var_lb, var_ub])
                break

            if arg in ('ub', 'ubound'):
                i += 1
                try:
                    arg = argv[i]
                except IndexError:
                    msg = 'the {} {}\'s'.format(var_name, _PREFIX_NAME[prefix])
                    msg += ' upper bound is not provided.'
                    raise cr.CRError(msg)

                try:
                    var_ub = float(arg)
                except ValueError:
                    # ub can be an equal-style variable
                    var_ub = lmp.extract_variable(arg, None, 0)
                    if var_ub is None:
                        msg = 'ub must be followed by an equal-style variable.'
                        raise cr.CRError(msg)
            else:
                i -= 1

            ctrl_map[var_name] = tuple([var_lb, var_ub])

        elif arg in ('ubound', 'ub'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'upper bound is not provided.'
                raise cr.CRError(msg)

            # being here means that this ctrl variable has no lower bound
            var_lb = None

            try:
                var_ub = float(arg)
            except ValueError:
                # lb & ub must be equal-style variable
                var_ub = lmp.extract_variable(arg, None, 0)
                if var_ub is None:
                    msg = 'ub must be followed by an equal-style variable.'
                    raise cr.CRError(msg)

            ctrl_map[var_name] = tuple([var_lb, var_ub])

        elif arg in ('population_mean', 'mean'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_mean is not provided.'
                raise cr.CRError(msg)

            if population_mean is None:
                population_mean = {}

            try:
                value = float(arg)
            except ValueError:
                value = lmp.extract_variable(arg, None, 0)

                if value is None:
                    msg = 'population_mean must be followed by an '
                    msg += 'equal-style variable.'
                    raise cr.CRError(msg)

            population_mean[var_name] = value

        elif arg in ('population_std', 'std'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_std is not provided.'
                raise cr.CRError(msg)

            if population_std is None:
                population_std = {}

            try:
                value = float(arg)
            except ValueError:
                value = lmp.extract_variable(arg, None, 0)

                if value is None:
                    msg = 'population_std must be followed by an '
                    msg += 'equal-style variable.'
                    raise cr.CRError(msg)

            population_std[var_name] = value

        elif arg in ('population_cdf', 'cdf'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_cdf is not provided.'

            if population_cdf is None:
                population_cdf = {}

            population_cdf[var_name] = arg

        elif arg in ('population_args', 'args'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_args is not provided.'

            if population_args is None:
                population_args = {}
                population_args[var_name] = []

            brackets = ('()', '{}', '[]', '(,)', '{,}',
                        '[,]', '(', ')', '[', ']', '{', '}')

            arg = arg.replace(' ', '')

            for b in brackets:
                if b in arg:
                    arg = arg.replace(b, '')

            if len(arg):
                arg = arg.split(',')

            for arg_ in arg:
                try:
                    value = int(arg_)
                except ValueError:
                    try:
                        value = float(arg_)
                    except ValueError:
                        msg = 'population_args must be followed by a '
                        msg += 'list or tuple of values(s).'
                        raise cr.CRError(msg)

                population_args[var_name].append(value)

        elif arg in ('population_loc', 'loc'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_loc is not provided.'

            if population_loc is None:
                population_loc = {}

            try:
                value = float(arg)
            except ValueError:
                value = lmp.extract_variable(arg, None, 0)

                if value is None:
                    msg = 'population_loc must be followed by an '
                    msg += 'equal-style variable.'
                    raise cr.CRError(msg)

            population_loc[var_name] = value

        elif arg in ('population_scale', 'scale'):
            i += 1
            try:
                arg = argv[i]
            except IndexError:
                msg = 'the {} {}\'s '.format(var_name, _PREFIX_NAME[prefix])
                msg += 'population_scale is not provided.'

            if population_scale is None:
                population_scale = {}

            try:
                value = float(arg)
            except ValueError:
                value = lmp.extract_variable(arg, None, 0)

                if value is None:
                    msg = 'population_scale must be followed by an '
                    msg += 'equal-style variable.'
                    raise cr.CRError(msg)

            population_scale[var_name] = value

        i += 1

    # Run the LAMMPS script
    lmp.command(cmd)

    if ctrl_map:
        if argument_counter == 1:
            var_name = arguments_map[0]
            msg = 'the variable "{}" is used for '.format(var_name)
            msg += 'controling the stability of the simulation to be '
            msg += 'bounded by lower and/or upper bound. It can not be '
            msg += 'used for the run length control at the same time.'
            raise cr.CRError(msg)

        if argument_counter == len(ctrl_map):
            var_name = arguments_map[0]
            msg = 'the variables "{}", '.format(var_name)
            for i in range(1, argument_counter - 1):
                var_name = arguments_map[i]
                msg = '"{}", '.format(var_name)
            var_name = arguments_map[-1]
            msg = 'and "{}" are used for '.format(var_name)
            msg += 'controling the stability of the simulation to be '
            msg += 'bounded by lower and/or upper bounds. They can not be '
            msg += 'used for the run length control at the same time.'
            raise cr.CRError(msg)

    def get_trajectory(step: int, args: dict) -> np.ndarray:
        """Get trajectory vector or array of values.

        Arguments:
            step (int): number of steps to run the simulation.
            args (dict): arguments necessary to get the trajectory.

        Returns:
            ndarray: trajectory
                for a single specified value, the values are stored as a
                vector. For multiple specified values, they are stored as rows
                in an array.

        """
        args['stop'] += step

        finalstep = args['stop'] // nevery * nevery
        if finalstep > args['stop']:
            finalstep -= nevery
        ncountmax = (finalstep - args['initialstep']) // nevery + 1
        args['initialstep'] = finalstep + nevery

        # Run the LAMMPS simulation
        cmd = 'run {}'.format(step)
        lmp.command(cmd)

        if ctrl_map:
            # trajectory array
            _ndim = argument_counter - len(ctrl_map)
            trajectory = np.empty((_ndim, ncountmax), dtype=np.float64)

            # argument index in the trajectory array
            _j = 0
            for j in range(argument_counter):
                var_name = arguments_map[j]
                if var_name in ctrl_map:
                    lb, ub = ctrl_map[var_name]
                    if lb and ub:
                        for _nstep in range(args['nstep'], args['nstep'] + ncountmax):
                            val = lmp.extract_fix('cr_fix', 0, 2, _nstep, j)
                            if val <= lb or val >= ub:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '{} {}). '.format(lb, ub)
                                msg += 'This run is unstable.'
                                raise cr.CRError(msg)
                        continue
                    elif lb:
                        for _nstep in range(args['nstep'], args['nstep'] + ncountmax):
                            val = lmp.extract_fix('cr_fix', 0, 2, _nstep, j)
                            if val <= lb:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '{} ...). '.format(lb)
                                msg += 'This run is unstable.'
                                raise cr.CRError(msg)
                        continue
                    elif ub:
                        for _nstep in range(args['nstep'], args['nstep'] + ncountmax):
                            val = lmp.extract_fix('cr_fix', 0, 2, _nstep, j)
                            if val >= ub:
                                msg = 'the "{}"\'s value = '.format(var_name)
                                msg += '{} is out of bound of ('.format(val)
                                msg += '... {}). '.format(ub)
                                msg += 'This run is unstable.'
                                raise cr.CRError(msg)
                        continue
                else:
                    for i, _nstep in enumerate(range(args['nstep'], args['nstep'] + ncountmax)):
                        trajectory[_j, i] = \
                            lmp.extract_fix('cr_fix', 0, 2, _nstep, j)
                    _j += 1
            args['nstep'] += ncountmax
            if _ndim == 1:
                return trajectory.squeeze()
            return trajectory

        if argument_counter == 1:
            trajectory = np.empty((ncountmax), dtype=np.float64)
            for i, _nstep in enumerate(range(args['nstep'], args['nstep'] + ncountmax)):
                trajectory[i] = lmp.extract_fix('cr_fix', 0, 1, _nstep, 0)
            args['nstep'] += ncountmax
            return trajectory

        trajectory = np.empty((argument_counter, ncountmax), dtype=np.float64)
        for j in range(argument_counter):
            for i, _nstep in enumerate(range(args['nstep'], args['nstep'] + ncountmax)):
                trajectory[j, i] = lmp.extract_fix('cr_fix', 0, 2, _nstep, j)
        args['nstep'] += ncountmax
        return trajectory

    p_mean = None
    p_std = None
    p_cdf = None
    p_args = None
    p_loc = None
    p_scale = None

    if argument_counter == 1:
        var_name = arguments_map[0]
        if population_mean is not None:
            p_mean = population_mean[var_name]
        if population_std is not None:
            p_std = population_std[var_name]
        if population_cdf is not None:
            p_cdf = population_cdf[var_name]
        if population_args is not None:
            p_args = population_args[var_name]
        if population_loc is not None:
            p_loc = population_loc[var_name]
        if population_scale is not None:
            p_scale = population_scale[var_name]
    else:
        if population_mean is not None:
            p_mean = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_mean:
                    p_mean.append(population_mean[var_name])
                else:
                    p_mean.append(None)
        if population_std is not None:
            p_std = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_std:
                    p_std.append(population_std[var_name])
                else:
                    p_std.append(None)
        if population_cdf is not None:
            p_cdf = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_cdf:
                    p_cdf.append(population_cdf[var_name])
                else:
                    p_cdf.append(None)
        if population_args is not None:
            p_args = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_args:
                    p_args.append(population_args[var_name])
                else:
                    p_args.append(None)
        if population_loc is not None:
            p_loc = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_loc:
                    p_loc.append(population_loc[var_name])
                else:
                    p_loc.append(None)
        if population_scale is not None:
            p_scale = []
            for i in range(argument_counter):
                var_name = arguments_map[i]
                if var_name in population_scale:
                    p_scale.append(population_scale[var_name])
                else:
                    p_scale.append(None)

    get_trajectory_args = {
        'stop': 0,
        'nstep': 0,
        'initialstep': 0,
    }

    try:
        msg = cr.run_length_control(
            get_trajectory=get_trajectory,
            get_trajectory_args=get_trajectory_args,
            number_of_variables=argument_counter - len(ctrl_map),
            initial_run_length=INITIAL_RUN_LENGTH,
            run_length_factor=RUN_LENGTH_FACTOR,
            maximum_run_length=MAX_RUN_LENGTH,
            maximum_equilibration_step=MAX_EQUILIBRATION_STEP,
            minimum_number_of_independent_samples=MINIMUM_NUMBER_OF_INDEPENDENT_SAMPLES,
            relative_accuracy=RELATIVE_ACCURACY,
            absolute_accuracy=ABSOLUTE_ACCURACY,
            population_mean=p_mean,
            population_standard_deviation=p_std,
            population_cdf=p_cdf,
            population_args=p_args,
            population_loc=p_loc,
            population_scale=p_scale,
            confidence_coefficient=CONFIDENCE,
            confidence_interval_approximation_method=UCL_METHOD,
            heidel_welch_number_points=cr._default._DEFAULT_HEIDEL_WELCH_NUMBER_POINTS,
            fft=cr._default._DEFAULT_FFT,
            test_size=cr._default._DEFAULT_TEST_SIZE,
            train_size=cr._default._DEFAULT_TRAIN_SIZE,
            batch_size=cr._default._DEFAULT_BATCH_SIZE,
            scale=cr._default._DEFAULT_SCALE_METHOD,
            with_centering=cr._default._DEFAULT_WITH_CENTERING,
            with_scaling=cr._default._DEFAULT_WITH_SCALING,
            ignore_end=cr._default._DEFAULT_IGNORE_END,
            number_of_cores=cr._default._DEFAULT_NUMBER_OF_CORES,
            si=cr._default._DEFAULT_SI,
            nskip=cr._default._DEFAULT_NSKIP,
            minimum_correlation_time=cr._default._DEFAULT_MINIMUM_CORRELATION_TIME,
            dump_trajectory=DUMP_TRAJECTORY,
            dump_trajectory_fp='kim_convergence_trajectory.edn',
            fp='return',
            fp_format='txt')
    except Exception as e:
        msg = '{}'.format(e)
        raise cr.CRError(msg)

    cmd = "variable run_var string ''"
    lmp.command(cmd)

    lmp.set_string_variable('run_var', msg)

    cmd = 'print "${run_var}"'
    lmp.command(cmd)

    cmd = "variable run_var delete"
    lmp.command(cmd)
