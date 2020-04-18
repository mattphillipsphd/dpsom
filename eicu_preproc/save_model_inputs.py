"""
Pre-processing functions for the T-DPSOM model, save data-set locally.
"""

import argparse
import parmap
import numpy as np
import os
from glob import glob
import pandas as pd
import h5py

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_normalized_data(data, patientid, mins, scales):
    if mins==None or scales==None:
        return ( data[data['patientunitstayid'] == patientid] ).drop(\
                ["patientunitstayid", "ts"], axis=1).fillna(0).values
    return ((data[data['patientunitstayid'] == patientid] - mins) /
            scales).drop(["patientunitstayid", "ts"], axis=1).fillna(0).values


def get_patient_n(patient, data_frame, data_frame_endpoint, max_n_step,
        mins_dynamic, scales_dynamic):
    time_series_all = []
    time_series_endpoint_all = []
    patient_data = get_normalized_data(data_frame, patient, mins_dynamic,
            scales_dynamic)
    patient_endpoint = data_frame_endpoint[\
            data_frame_endpoint['patientunitstayid'] == patient].drop(\
            ["patientunitstayid", "ts"], axis=1)
    patient_endpoint = patient_endpoint[['full_score_1', 'full_score_6',
        'full_score_12', 'full_score_24',
        'hospital_discharge_expired_1', 'hospital_discharge_expired_6',
        'hospital_discharge_expired_12', 'hospital_discharge_expired_24',
        'unit_discharge_expired_1', 'unit_discharge_expired_6',
        'unit_discharge_expired_12', 'unit_discharge_expired_24']]\
                .fillna(0).values

    if max_n_step > 0:
        time_series = patient_data[ :max_n_step ]
        time_series_endpoint = patient_endpoint[ :max_n_step ]
    else:
        max_n_step = -max_n_step
        time_series = patient_data[len(patient_data) - max_n_step: \
                len(patient_data)]
        time_series_endpoint = patient_endpoint[len(patient_data)-max_n_step: \
                len(patient_data)]
    time_series_all.append(time_series)
    time_series_endpoint_all.append(time_series_endpoint)

    return np.array(time_series_all), np.array(time_series_endpoint_all)


def parmap_batch_generator(data_total, endpoints_total, mins_dynamic,
        scales_dynamic, max_n_step):
    time_series_all = []
    time_series_endpoint_all = []
    for p in range(len(data_total)):
        print(p)
        path = data_total[p]
        path_endpoint = endpoints_total[p]
        data_frame = pd.read_hdf(path).fillna(0)
        data_frame_endpoint = pd.read_hdf(path_endpoint).fillna(0)
        assert not data_frame.isnull().values.any(), "No NaNs allowed"
        assert not data_frame_endpoint.isnull().values.any(), "No NaNs allowed"
        patients = data_frame.patientunitstayid.unique()

        temp = parmap.map(get_patient_n, patients, data_frame,
                data_frame_endpoint, max_n_step, mins_dynamic,
                scales_dynamic)

        data = []
        labels = []
        for a in range(len(temp)):
            for b in range(len(temp[a][1])):
                labels.append(temp[a][1][b])
                data.append(temp[a][0][b])
        data = np.array(data)
        labels = np.array(labels)
        time_series_all.extend(data)
        time_series_endpoint_all.extend(labels)

    return time_series_all, time_series_endpoint_all


def main(cfg):
    # path of the preprocessed data
    data_total = glob( pj(HOME, "Datasets/eicu-2.0/time_grid/batch_*.h5") )

    # path of the labels of the preprocessed data
    endpoints_total = glob( pj(HOME, "Datasets/eicu-2.0/labels/batch_*.h5") )

    normalization_path = pj(HOME, "Datasets/eicu-2.0/time_grid/" \
            "normalization_values.h5")
    if pe(normalization_path):
        # path of the labels of the mins
        mins_dynamic = pd.read_hdf(normalization_path , "mins_dynamic")

        # path of the labels of the scales
        scales_dynamic = pd.read_hdf(normalization_path, "scales_dynamic")
        has_normalization = True
    else:
        mins_dynamic,scales_dynamic = None, None
        print("Warning: using non-normalized data")
        has_normalization = False

    # *************************************************************************

    # Create numpy arrays with the last max_n_step time-steps of each
    # time-series.
    max_n_step = cfg["max_n_step"]
    data, labels = parmap_batch_generator(data_total, endpoints_total,
            mins_dynamic, scales_dynamic, max_n_step=max_n_step)
    l = np.array(labels)
    d = np.array(data)
    stub = "eICU_data"
    if max_n_step<0:
        stub += "_b"
    else:
        stub += "_e"
    stub += "%d" % np.abs(max_n_step)
    if not has_normalization:
        stub += "_nonorm"
    output_path = pj(HOME, "Datasets/eicu-2.0", stub+".csv")
    hf = h5py.File(output_path, 'w')
    hf.create_dataset('x', data=d)
    hf.create_dataset('y', data=l)
    hf.close()
    print("Wrote data to %s" % output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-n-step", type=int, default=-72,
            help="If negative, the last -n time steps are used.  If positive, "\
                    "the first n time steps are used.")
    cfg = vars( parser.parse_args() )
    main(cfg)
