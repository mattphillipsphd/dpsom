"""
Create a single csv file with all the static info in it
"""

import argparse
import numpy as np
import os
import pandas as pd

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def main(cfg):
    data_supdir = os.path.abspath( cfg["data_supdir"] )
    static_dir = pj(data_supdir, "static_pts")
    paths = [pj(static_dir,f) for f in os.listdir(static_dir)]
    df = pd.read_csv(paths[0])
    for path in paths[1:]:
        df = df.append( pd.read_csv(path) )
    df.to_csv( pj(data_supdir, "static.csv") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/EHRs/eICU"))
    cfg = vars( parser.parse_args() )
    main(cfg)

