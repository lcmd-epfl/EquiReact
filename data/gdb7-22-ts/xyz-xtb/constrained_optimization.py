#!/usr/bin/env python3

import random
import numpy as np
import time
from glob import glob
import os
import pysisyphus
import itertools
from pysisyphus.calculators import XTB
from pysisyphus.dynamics.helpers import get_mb_velocities_for_geom
from pysisyphus.helpers import geom_loader, imag_modes_from_geom
from pysisyphus.helpers_pure import eigval_to_wavenumber
from pysisyphus.testing import using
from pysisyphus.tsoptimizers import TRIM, RSIRFOptimizer, RSPRFOptimizer
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.constants import BOHR2ANG
import ray


def grouper_it(n, iterable):
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)


def xyz2geom(
    filename, constrain
):  # constrains : [["atom", 4], ["atom",6], ["atom", 11], ["atom",12]]
    geom_kwargs = {
        "coord_type": "redund",
    }
    if constrain is not None:
        geom_kwargs["coord_kwargs"] = {
            "constrain_prims": constrain,
        }
    geom = geom_loader(filename, **geom_kwargs)
    calc = XTB(pal=8, charge=0, mult=1)
    try:
        geom.set_calculator(calc)
        e = geom.energy
    except Exception as m:
        print("Could not converge XTB calculation, warning!")
    return geom


def geom2xyz(geom, path):
    atoms = geom.atoms
    coord_list = geom.coords3d * BOHR2ANG
    assert len(atoms) == len(coord_list)
    comment = str(path)
    coord_fmt = "{: 03.8f}"
    line_fmt = "{:>3s} " + " ".join(
        [
            coord_fmt,
        ]
        * 3
    )
    body = [line_fmt.format(a, *xyz) for a, xyz in zip(atoms, coord_list)]
    body = "\n".join(body)
    f = open(path, "w")
    print("{0}\n{1}\n{2}".format(len(atoms), comment, body), file=f)
    f.close()
    return geom


@ray.remote
def optimize_min(i):
    constrain = [
        ["atom", 0],
    ]
    try:
        print(f"Constrained optimization of {i}")
        geom = xyz2geom(i, constrain)
        initialcoords = geom.coords
        opt_kwargs = {
            "thresh": "gau_loose",
            "line_search": True,
            "gdiis": True,
            "hessian_xtb": True,
            "dump": False,
            "overachieve_factor": 1.0,
            "max_cycles": 1000,
            "adapt_step_func": True,
        }
        opt = RFOptimizer(geom, **opt_kwargs)
        opt.run()
        opti = "opt_min/{0}_opt.xyz".format(i[:-4])
        print(f"Wrote optimized geom to {opti}")
        geom2xyz(geom, opti)
        done = "OK"
    except Exception as m:
        done = m
    return done


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


ray.init()
directory = os.path.join(".", "*.xyz")
it_full = glob(directory)
it_random = randomly(it_full)
tic = time.perf_counter()
batches = grouper_it(24, it_random)
for i in batches:
    done = ray.get([optimize_min.remote(j) for j in i])
    print(done)
    toc = time.perf_counter()
    print(f"Time for batch: {toc - tic:0.4f} seconds")
