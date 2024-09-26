from typing import Optional, Sequence

# A relative half-width requirement or the accuracy parameter. Target value
# for the ratio of halfwidth to sample mean. If n_variables > 1,
# relative_accuracy can be a scalar to be used for all variables or a 1darray
# of values of size n_variables.
# For cells, we can only use a relative accuracy for all non-zero variables.
# The last three variables, however, correspond to the tilt factors of the orthogonal cell (see npt.lammps which are
# expected to fluctuate around zero. For these, we should use an absolute accuracy instead.
RELATIVE_ACCURACY: Sequence[Optional[float]] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, None, None, None]
ABSOLUTE_ACCURACY: Sequence[Optional[float]] = [None, None, None, None, None, None, 0.01, 0.01, 0.01]
