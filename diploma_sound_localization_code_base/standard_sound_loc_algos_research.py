import math

import numpy as np
import pandas as pd
import csv
from itertools import combinations
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import sounddevice as snd
import argparse
from scipy.io import wavfile
import os

mic_array_audio_files_with_low_SNR_dir_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/59c3a7f2_source_3_rt60_2.0_SNR_60/"

mic_0_audio_file_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/59c3a7f2_source_0_rt60_2.0_SNR_10/59c3a7f2_source_0_mic_0_rt60_2.0_SNR_10.wav"
mic_1_audio_file_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/59c3a7f2_source_0_rt60_2.0_SNR_10/59c3a7f2_source_0_mic_1_rt60_2.0_SNR_10.wav"
mic_2_audio_file_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/59c3a7f2_source_0_rt60_2.0_SNR_10/59c3a7f2_source_0_mic_2_rt60_2.0_SNR_10.wav"

sources_params_file_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/sources_params.csv"
mics_params_file_path = "C:/University/Diploma Sound Localization/diploma_sound_localization_data/microphones_params.csv"

sources_params_df = pd.read_csv(sources_params_file_path)
mics_params_df = pd.read_csv(mics_params_file_path)

print("Sources params:")
print(sources_params_df)

print("Microphones params:")
print(mics_params_df)

fs = 16000
c = 343
nfft = 256  # FFT size
freq_range = [300, 3500]

algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS']
spatial_resp = dict()

X = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
X = X.transpose([2, 1, 0])

# loop through algos
for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=2, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_range=freq_range)

    # store spatial response
    if algo_name is 'FRIDA':
        spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
    else:
        spatial_resp[algo_name] = doa.grid.values

    # normalize
    min_val = spatial_resp[algo_name].min()
    max_val = spatial_resp[algo_name].max()
    spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)

def parse_audio_file_name_to_params(audio_file_name):
    params = audio_file_name.split("_")[1:]
    return {params[i]: params[i + 1] for i in range(0, len(params), 2)}

def calculate_distance_from_samples_and_velocity(samples_num, fs, v):
    return samples_num / fs * v

def calculate_distance_between_2d_points(first_point, second_point):
    return math.sqrt((first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2)

def calculate_tdoa_mae_for_mic_array_audio_signals(mic_array_audio_files_dir_path):
    dir_name = (mic_array_audio_files_dir_path.split("/"))[-2]
    dir_params = parse_audio_file_name_to_params(dir_name)

    source_params = sources_params_df.loc[sources_params_df["s_num"] == int(dir_params["source"])]
    source_coords = ((source_params["x"]).values[0], (source_params["y"]).values[0])
    mics_audio_files_paths = os.listdir(mic_array_audio_files_dir_path)

    mics_num = len(mics_audio_files_paths)
    mae_sum = 0

    mics_audio_files_paths_pairs = list(combinations(mics_audio_files_paths, 2))
    print(mics_audio_files_paths_pairs)

    for mic_audio_files_paths_pair in mics_audio_files_paths_pairs:
        first_fs, first_mic_audio_signal = wavfile.read(mic_array_audio_files_dir_path + mic_audio_files_paths_pair[0])
        second_fs, second_mic_audio_signal = wavfile.read(mic_array_audio_files_dir_path + mic_audio_files_paths_pair[1])
        actual_tdoa = pra.sync.tdoa(first_mic_audio_signal, second_mic_audio_signal) / fs
        print("Actual TDOA: "+ str(actual_tdoa))

        first_mic_params = parse_audio_file_name_to_params(mic_audio_files_paths_pair[0])
        second_mic_params = parse_audio_file_name_to_params(mic_audio_files_paths_pair[1])
        first_mic_coords = mics_params_df.loc[mics_params_df["m_num"] == int(first_mic_params["mic"])]
        second_mic_coords = mics_params_df.loc[mics_params_df["m_num"] == int(second_mic_params["mic"])]

        first_mic_to_source_dist = calculate_distance_between_2d_points(
            tuple([first_mic_coords["x"].values[0], first_mic_coords["y"].values[0]]),
            tuple([(source_params["x"]).values[0], (source_params["y"]).values[0]])
        )

        second_mic_to_source_dist = calculate_distance_between_2d_points(
            tuple([second_mic_coords["x"].values[0], second_mic_coords["y"].values[0]]),
            tuple([(source_params["x"]).values[0], (source_params["y"]).values[0]])
        )

        expected_tdoa = (second_mic_to_source_dist - first_mic_to_source_dist) / c

        print("Expected TDOA: " + str(expected_tdoa))

        mae_sum += abs(expected_tdoa - actual_tdoa)

    return mae_sum / mics_num


if __name__ == '__main__':
    #print(pra.sync.tdoa(mic0_audio_signal, mic2_audio_signal) / fs)

    print("TDOA MAE = " + str(calculate_tdoa_mae_for_mic_array_audio_signals(mic_array_audio_files_with_low_SNR_dir_path)))

