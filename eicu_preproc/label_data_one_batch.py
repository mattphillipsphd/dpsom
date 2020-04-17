"""
Cluster dispatcher for label generation
"""

import os
import argparse
import pickle
import sys

import pandas as pd

import classes.dynamic_endpoints as eicu_dynamic_tf

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def label_data_one_batch(configs):
    first_write = True
    batch_id = configs["batch_id"]

    with open(configs["pid_batch_file"], 'rb') as fp:
        obj = pickle.load(fp)
        batch_to_lst = obj["batch_to_lst"]
        batches = list(sorted(batch_to_lst.keys()))
        batch_idxs = batch_to_lst[batch_id]        

    print("Dispatched batch {} with {} patients".format(batch_id,
        len(batch_idxs)))

    for pidx, pid in enumerate(batch_idxs):

        if (pidx+1) % 10 == 0:
            print("Progress in batch {}: {}/{}".format(batch_id, pidx+1,
                len(batch_idxs)))

        dynamic_extractor = eicu_dynamic_tf.DynamicEndpointExtractor()
        df_pat = pd.read_hdf(configs["input_patient_table"], mode='r',
                where="patientunitstayid={}".format(pid))
        df_imputed = pd.read_hdf(os.path.join(configs["imputed_data_dir"],
            "batch_{}.h5".format(batch_id)), mode='r',
            where="patientunitstayid={}".format(pid))
        df_dynamic_endpoints = dynamic_extractor.transform(df_imputed, df_pat,
                pid=pid)

        if first_write:
            df_dynamic_endpoints.to_hdf(pj(\
                    configs["output_dynamic_endpoint_dir"],
                    "batch_{}.h5".format(batch_id)), configs["output_dset_id"],
                    append=False, data_columns=["patientunitstayid"], mode='w',
                    format="table", complevel=configs["hdf_comp_level"],
                    complib=configs["hdf_comp_alg"])
        else:
            df_dynamic_endpoints.to_hdf(pj(\
                    configs["output_dynamic_endpoint_dir"],
                    "batch_{}.h5".format(batch_id)), configs["output_dset_id"],
                    append=True, data_columns=["patientunitstayid"], mode='a',
                    format="table", complevel=configs["hdf_comp_level"],
                    complib=configs["hdf_comp_alg"])

        first_write = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--imputed_data_dir",
            default=pj(HOME, "Datasets/eicu-2.0/time_grid"),
            help="Imputed data dir")
    parser.add_argument("--input_patient_table",
            default=pj(HOME, "Datasets/eicu-2.0/hdf/patient.h5"),
            help="Input patient table path") 
    parser.add_argument("--pid_batch_file",
            default=pj(HOME, "Datasets/eicu-2.0/patient_batches.pickle"),
            help="Specify the map from PIDs to batches") 

    # Output paths
    parser.add_argument("--output_dynamic_endpoint_dir",
            default=pj(HOME, "Datasets/eicu-2.0/labels"),
            help="Output directory for dynamic endpoints") 
    parser.add_argument("--log_dir",
            default=pj(HOME, "Datasets/eicu-2.0/logs"),
            help="Logging directory")

    # Arguments
    parser.add_argument("--output_dset_id",
            default='data',
            help="Generic HDF dset ID")
    parser.add_argument("--hdf_comp_level", type=int,
            default=5,
            help="HDF compression level")
    parser.add_argument("--hdf_comp_alg",
            default="blosc:lz4",
            help="HDF compression algorithm")
    parser.add_argument("--batch_id", type=int,
            default=0,
            help="Batch ID to process")

    parser.add_argument("--run_mode",
            default="INTERACTIVE",
            help="Running mode, interactive or on cluster?")    

    args = parser.parse_args()
    configs = vars(args)

    if configs["run_mode"]=="CLUSTER":
        sys.stdout=open(pj(configs["log_dir"],
            "labels_batch_{}.stdout".format(configs["batch_id"])),'w')
        sys.stderr=open(pj(configs["log_dir"],
            "labels_batch_{}.stderr".format(configs["batch_id"])),'w')    

    label_data_one_batch(configs)
