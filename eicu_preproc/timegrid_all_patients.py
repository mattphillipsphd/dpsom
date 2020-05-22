"""
Dispatch script for imputation/time-gridding
"""

import subprocess
import argparse
import pickle
import os
import sys

import functions.util_filesystem as mlhc_fs

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def timegrid_all_patients(configs):
    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=1
    n_compute_hours=configs["nhours"]
    compute_script_path=configs["compute_script_path"]

    with open(configs["patient_batch_path"],'rb') as fp:
        obj=pickle.load(fp)
        batch_to_lst=obj["batch_to_lst"]
        batches=list(sorted(batch_to_lst.keys()))

    create_pars = []
    for p in ["create_static", "create_dynamic", "create_async"]:
        if configs[p]:
            create_pars.append("--" + p + " True")

    for batch_idx in batches:
        print("Dispatching imputation for batch {}".format(batch_idx))
        job_name = "impute_batch_{}".format(batch_idx)
        log_result_file = pj( configs["log_base_dir"],
                "impute_batch_{}_RESULT.txt".format(batch_idx) )
        mlhc_fs.delete_if_exist(log_result_file)

        cmd_line=" ".join(["python3", configs["compute_script_path"],
            "--run_mode INTERACTIVE", "--batch_id {}".format(batch_idx)] \
                    + create_pars)

        assert(" rm " not in cmd_line)
        job_index+=1

        if not configs["dry_run"]:
            subprocess.call([cmd_line], shell=True)
        else:
            print("Generated cmd line: [{}]".format(cmd_line))

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--patient_batch_path",
            default=pj(HOME, "Datasets/eicu-2.0/patient_batches.pickle"),
            help="The path of the PID-Batch map") 
    parser.add_argument("--compute_script_path",
            default=pj(HOME, "Repos/mattphillipsphd/dpsom/eicu_preproc" \
                    "/timegrid_one_batch.py"),
            help="Script to dispatch")

    # Output paths
    parser.add_argument("--log_base_dir",
            default=pj(HOME, "Datasets/eicu-2.0/logs"),
            help="Log base directory") 

    # Parameters
    parser.add_argument("--dry_run", action="store_true",
            default=False,
            help="Dry run, do not generate any jobs")
    parser.add_argument("--mem_in_mbytes", type=int,
            default=5000,
            help="Number of MB to request per script")
    parser.add_argument("--nhours", type=int,
            default=4,
            help="Number of hours to request")
    parser.add_argument("--create_static", action="store_true")
    parser.add_argument("--create_dynamic", action="store_true")
    parser.add_argument("--create_async", action="store_true")

    args=parser.parse_args()
    configs=vars(args)
    
    timegrid_all_patients(configs)
    
