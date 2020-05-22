"""
Generates the time-gridded data
"""

import argparse
import os
import os.path
import csv
import time
import timeit
import psutil
import pickle
import sys
import json

import pandas as pd

import matplotlib as mpl
mpl.use("PDF")

import classes.imputer as eicu_tf_impute
import classes.static_extractor as eicu_static_tf

import functions.util_io as mlhc_io

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def timegrid_one_batch(configs):
    batch_id=configs["batch_id"]    

    with open(configs["pid_batch_file"], 'rb') as fp:
        obj = pickle.load(fp)
        batch_to_lst = obj["batch_to_lst"]
        batches = list(sorted(batch_to_lst.keys()))
        batch_idxs = batch_to_lst[batch_id]    
    
    first_write = True
    create_static = configs["create_static"]
    create_dynamic = configs["create_dynamic"]
    create_async = configs["create_async"]
    print("Dispatched batch {} with {} patients".format(batch_id,
        len(batch_idxs)))

    for pidx, pid in enumerate(batch_idxs):

        if (pidx+1) % 10 == 0:
            print("Progress in batch {}: {}/{}".format(batch_id, pidx+1,
                len(batch_idxs)))

        if create_static:
            static_extractor = eicu_static_tf.StaticExtractor()
            df_pat = pd.read_hdf(configs["input_patient_table"], mode='r',
                    where="patientunitstayid={}".format(pid))
            df_adm = pd.read_hdf(configs["input_admission_table"], mode='r',
                    where="patientunitstayid={}".format(pid))
            df_aav = pd.read_hdf(configs["input_apache_aps_var_table"],
                    mode='r', where="patientunitstayid={}".format(pid))
            df_apr = pd.read_hdf(configs["input_apache_patient_result_table"],
                    mode='r', where="patientunitstayid={}".format(pid))
            df_apv = pd.read_hdf(configs["input_apache_pred_var_table"],
                    mode='r', where="patientunitstayid={}".format(pid))
            df_static = static_extractor.transform(df_pat, df_adm, df_aav,
                    df_apr, df_apv, pid=pid)

        if create_dynamic or create_async:
            lab_vars = []

            with open(configs["selected_lab_vars"], 'r') as fp:
                csv_fp = csv.reader(fp, delimiter='\t')
                next(csv_fp)
                for lab_name in csv_fp:
                    lab_vars.append(lab_name[0].strip())

            grid_model = eicu_tf_impute.Timegridder()
            grid_model.set_selected_lab_vars(lab_vars)

            quantile_fp = open(configs["quantile_dict"], mode='r')
            var_quantile_dict = json.load(quantile_fp)
            grid_model.set_quantile_dict(var_quantile_dict)
            quantile_fp.close()    

            df_lab = pd.read_hdf(configs["input_lab_table"], mode='r',
                    where="patientunitstayid={}".format(pid))
            df_vs = pd.read_hdf(configs["input_vital_periodic_table"],
                    mode='r', where="patientunitstayid={}".format(pid))
            df_avs = pd.read_hdf(configs["input_vital_aperiodic_table"],
                    mode='r', where="patientunitstayid={}".format(pid))
            if create_dynamic:
                df_out = grid_model.transform(df_lab, df_vs, df_avs, pid=pid)
            if create_async:
                df_async = grid_model.save_async(df_lab, df_vs, df_avs,
                        pid=pid, timegrid_step_mins=1)

        append = not first_write
        mode = 'a' if first_write else 'w'

        if create_dynamic:
            df_out.to_hdf(os.path.join(configs["output_dynamic_dir"],
                "batch_{}.h5".format(batch_id)), configs["output_dset_id"],
                append=append, data_columns=["patientunitstayid"],
                mode=mode, format="table",
                complevel=configs["hdf_comp_level"],
                complib=configs["hdf_comp_alg"])

        if create_async:
            df_async.to_hdf(os.path.join(configs["output_async_dir"],
                "batch_{}.h5".format(batch_id)), configs["output_dset_id"],
                append=append, data_columns=["patientunitstayid"],
                mode=mode, format="table",
                complevel=configs["hdf_comp_level"],
                complib=configs["hdf_comp_alg"])

        if create_static:
            df_static.to_hdf(os.path.join(configs["output_static_dir"],
                "batch_{}.h5".format(batch_id)), configs["output_dset_id"],
                append=append, data_columns=["patientunitstayid"],
                mode=mode, format="table",
                complevel=configs["hdf_comp_level"],
                complib=configs["hdf_comp_alg"])

        first_write = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Location of the input tables
    parser.add_argument("--input_patient_table",
            default=pj(HOME, "Datasets/eicu-2.0/hdf/patient.h5"),
            help="Location of the patient table")
    parser.add_argument("--input_admission_table",
            default=pj(HOME, "Datasets/eicu-2.0/hdf/admissionDx.h5"),
            help="Location of the admission table") 
    parser.add_argument("--input_apache_aps_var_table",
            default=pj(HOME, "Datasets/eicu-2.0/hdf/apacheApsVar.h5"),
                help="Location of the Apache APS Var table") 
    parser.add_argument("--input_apache_patient_result_table",
        default=pj(HOME, "Datasets/eicu-2.0/hdf/apachePatientResult.h5"),
        help="Location of the APACHE patient result table") 
    parser.add_argument("--input_apache_pred_var_table",
        default=pj(HOME, "Datasets/eicu-2.0/hdf/apachePredVar.h5"),
        help="Location of the APACHE pred var table") 
    parser.add_argument("--input_lab_table",
        default=pj(HOME, "Datasets/eicu-2.0/hdf/lab.h5"),
        help="Location of the lab table") 
    parser.add_argument("--input_vital_periodic_table",
        default=pj(HOME, "Datasets/eicu-2.0/hdf/vitalPeriodic.h5"),
        help="Location of the vital periodic table") 
    parser.add_argument("--input_vital_aperiodic_table",
        default=pj(HOME, "Datasets/eicu-2.0/hdf/vitalAperiodic.h5"),
            help="Location of the vital aperiodic table") 

    # Location of some meta-files
    parser.add_argument("--selected_pid_list",
        default=pj(HOME, "Datasets/eicu-2.0/included_pid_stays.txt"),
        help="Specify the lists of PIDs to use") 
    parser.add_argument("--pid_batch_file",
            default=pj(HOME, "Datasets/eicu-2.0/patient_batches.pickle"),
            help="Specify the map from PIDs to batches") 
    parser.add_argument('--selected_lab_vars',
            default=pj(HOME, "Datasets/eicu-2.0/included_lab_variables.txt"),
            help="Specify the file with the list of lab variables to use") 
    parser.add_argument("--quantile_dict",
            default=pj(HOME, "Datasets/eicu-2.0/var_quantiles.json"),
            help="Precomputed data quantiles in the eICU data-set that can be"\
                    " used to remove outliers")

    # Output paths

    parser.add_argument("--output_static_dir",
            default=pj(HOME, "Datasets/eicu-2.0/static"),
            help="Specify the output directory for static data") 
    parser.add_argument("--output_dynamic_dir",
            default=pj(HOME, "Datasets/eicu-2.0/time_grid"),
            help="Specify the output directory for dynamic time grid data")
    parser.add_argument("--output_async_dir",
            default=pj(HOME, "Datasets/eicu-2.0/async_data"),
            help="Specify the output directory for dynamic non-time grid data")
    parser.add_argument("--log_dir",
            default=pj(HOME, "Datasets/eicu-2.0/logs"),
            help="Logging directory") 

    # PARAMETERS
    
    # What output should be created?
    parser.add_argument('--create_static',
            default=False,
            help="Should static variables be output?")
    parser.add_argument("--create_dynamic",
            default=False,
            help="Should dynamic variables be output?")
    parser.add_argument("--create_async",
            default=False,
            help="Should dynamic variables without timegridding be output?")
    parser.add_argument("--batch_id", type=int,
            default=0,
            help="Batch index to process in this script")
    parser.add_argument("--run_mode",
            default="INTERACTIVE",
            help="Running mode, interactive or on cluster?")

    # Various arguments
    parser.add_argument("--hdf_comp_level",
            default=5,
            help="HDF compression level to output")
    parser.add_argument("--hdf_comp_alg",
            default="blosc:lz4",
            help="HDF compression algorithm to use")
    parser.add_argument("--output_dset_id",
            default="data",
            help="Data set name to use for all data-set")

    parser.add_argument("--debug", action="store_true",
            default=False,
            help="Debugging mode")

    args = parser.parse_args()
    configs = vars(args)

    if configs["run_mode"]=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],
            "time_grid_batch_{}.stdout".format(configs["batch_id"])),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],
            "time_grid_batch_{}.stderr".format(configs["batch_id"])),'w')

    timegrid_one_batch(configs)
