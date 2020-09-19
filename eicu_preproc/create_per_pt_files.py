"""
Create single files, per patient, for the data
"""

import argparse
import numpy
import os
import pandas as pd

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def main(cfg):
    data_supdir = os.path.abspath( cfg["data_supdir"] )
    batches_dir = pj(data_supdir, cfg["batches_subdir"])
    pts_dir = pj(data_supdir, cfg["batches_subdir"]+"_pts")
    if not pe(pts_dir):
        os.makedirs(pts_dir)
    print("Processing batches...")
    for f in os.listdir(batches_dir):
        if not f.endswith(".h5"):
            continue
        df = pd.read_hdf( pj(batches_dir,f) )
        convert_d = {"patientunitstayid" : "int64", "ts" : "int64"}
        df = df.astype(convert_d)
        pt_ids = df.patientunitstayid.unique()
        for pt_id in pt_ids:
            df_pt = df[ df.patientunitstayid==pt_id ]
            df_pt.to_csv( pj(pts_dir, str(pt_id)+".csv") )
    print("...Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-supdir", type=str,
            default=pj(HOME, "Datasets/EHRs/eICU"))
    parser.add_argument("-b", "--batches-subdir", type=str,
            default="time_grid")
    cfg = vars( parser.parse_args() )
    main(cfg)

