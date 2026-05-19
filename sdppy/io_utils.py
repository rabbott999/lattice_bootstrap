import numpy as np
from mpmath import mp

def write_array(filename, arr):
    with open(filename, 'w') as f:
        f.write(" ".join(map(str, arr.shape)))
        f.write("\n\n")

        for x in arr.ravel():
            f.write(f"{x.real} {x.imag}\n")

def read_mp_array(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

        shape = [int(x) for x in lines[0].split(" ")]

        def parse_row(row):
            return mp.mpf(row)

        values = [parse_row(row) for row in lines[1:]]
        assert len(values) == np.prod(shape)

    arr = np.array(values, dtype=object)
    arr = arr.reshape(shape)
    return arr

def mpf_array_str(a):
    if isinstance(a, mp.mpf):
        return mp.nstr(a, mp.dps + 3)
    return [mpf_array_str(x) for x in a]

def file_exists(filename):
    try:
        open(filename, 'r')
        return True
    except:
        return False
