import numpy as np
import os
import pickle
from FeatureExtraction import extract_all_features

#bigger: win_length=1024, hop_length=512, n_bands=6, n_bins=72, n_mfcc=20
#ultrasmall: win_length=2048, hop_length=2048, n_bands=3, n_bins=3, n_mfcc=3)
#small: win_length=1024, hop_length=512, n_bands=3, n_bins=3, n_mfcc=3)
#normal: win_length=2048, hop_length=2048
#window = 44100//2

win_length=1024
hop_length=512

def preparing_features():
    data_dir = '../../../Analysis/Note_collector'

    #########
    data = open(os.path.normpath('/'.join([data_dir, 'RealPianoAllNotes.pickle'])), 'rb')
    Z = pickle.load(data)
    real_piano, real_features = extract_all_features(Z, type='real', win_length=win_length, hop_length=hop_length, n_bands=6, n_bins=72, n_mfcc=20)

    data = open(os.path.normpath('/'.join([data_dir, 'DiskPianoAllNotes.pickle'])), 'rb')
    Z = pickle.load(data)
    disk_piano, disk_features = extract_all_features(Z, type='disk', win_length=win_length, hop_length=hop_length, n_bands=6, n_bins=72, n_mfcc=20)

    data = open(os.path.normpath('/'.join([data_dir, 'DigitalPianoAllNotes.pickle'])), 'rb')
    Z = pickle.load(data)
    digital_piano, digital_features = extract_all_features(Z, type='digital', win_length=win_length, hop_length=hop_length, n_bands=6, n_bins=72, n_mfcc=20)

    AllFeatures = {'real_piano': real_piano, 'real_features': real_features,
                   'disk_piano': disk_piano, 'disk_features': disk_features,
                   'digital_piano': digital_piano, 'digital_features': digital_features}
    
    file_data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_aligned.pickle'])), 'wb')
    pickle.dump(AllFeatures, file_data)
    file_data.close()



def preparing_features_diff():
    data_dir = '../../../Analysis/Note_collector'

    #differences
    data = open(os.path.normpath('/'.join([data_dir, 'RealPianosDifferences2.pickle'])), 'rb')
    Z = pickle.load(data)
    real_piano_d, real_features_d, real_vels = [], [], []
    for i in range(len(Z['piano'])):
        real_piano_d.append(Z['piano'][i])
        real_features_d.append([Z['diff_cA2'][i][0], Z['diff_cA2'][i][1], Z['diff_cA2'][i][2],
                                Z['diff_cA3'][i][0], Z['diff_cA3'][i][1], Z['diff_cA3'][i][2],
                                Z['diff_rA2'][i][0], Z['diff_rA2'][i][1], Z['diff_rA2'][i][2],
                                Z['diff_rA3'][i][0], Z['diff_rA3'][i][1], Z['diff_rA3'][i][2]])
        real_vels.append([Z['diff_vel_ch_A2'][i], Z['diff_vel_ch_A3'][i],
                         Z['diff_vel_rep_A2'][i], Z['diff_vel_rep_A3'][i]])

    data = open(os.path.normpath('/'.join([data_dir, 'DiskPianosDifferences2.pickle'])), 'rb')
    Z = pickle.load(data)
    disk_piano_d, disk_features_d, disk_vels = [], [], []
    for i in range(len(Z['piano'])):
        disk_piano_d.append(Z['piano'][i])
        disk_features_d.append([Z['diff_cA2'][i][0], Z['diff_cA2'][i][1], Z['diff_cA2'][i][2],
                                Z['diff_cA3'][i][0], Z['diff_cA3'][i][1], Z['diff_cA3'][i][2],
                                Z['diff_rA2'][i][0], Z['diff_rA2'][i][1], Z['diff_rA2'][i][2],
                                Z['diff_rA3'][i][0], Z['diff_rA3'][i][1], Z['diff_rA3'][i][2]])

        disk_vels.append([Z['diff_vel_ch_A2'][i], Z['diff_vel_ch_A3'][i],
                      Z['diff_vel_rep_A2'][i], Z['diff_vel_rep_A3'][i]])

    data = open(os.path.normpath('/'.join([data_dir, 'DigitalPianosDifferences2.pickle'])), 'rb')
    Z = pickle.load(data)
    digital_piano_d, digital_features_d, digital_vels = [], [], []
    for i in range(len(Z['piano'])):
        digital_piano_d.append(Z['piano'][i])
        digital_features_d.append([Z['diff_cA2'][i][0], Z['diff_cA2'][i][1], Z['diff_cA2'][i][2],
                                Z['diff_cA3'][i][0], Z['diff_cA3'][i][1], Z['diff_cA3'][i][2],
                                Z['diff_rA2'][i][0], Z['diff_rA2'][i][1], Z['diff_rA2'][i][2],
                                Z['diff_rA3'][i][0], Z['diff_rA3'][i][1], Z['diff_rA3'][i][2]])

        digital_vels.append([Z['diff_vel_ch_A2'][i], Z['diff_vel_ch_A3'][i],
                      Z['diff_vel_rep_A2'][i], Z['diff_vel_rep_A3'][i]])

    data = open(os.path.normpath('/'.join([data_dir, 'DiskPianosDifferencesAddition2.pickle'])), 'rb')
    Z = pickle.load(data)
    disk_piano_d_addition, disk_features_d_addition, disk_vels_addition = [], [], []
    for i in range(len(Z['piano'])):
        disk_piano_d_addition.append(Z['piano'][i])
        disk_features_d_addition.append([Z['diff_cA2'][i][0], Z['diff_cA2'][i][1], Z['diff_cA2'][i][2],
                                Z['diff_cA3'][i][0], Z['diff_cA3'][i][1], Z['diff_cA3'][i][2],
                                Z['diff_rA2'][i][0], Z['diff_rA2'][i][1], Z['diff_rA2'][i][2],
                                Z['diff_rA3'][i][0], Z['diff_rA3'][i][1], Z['diff_rA3'][i][2]])

        disk_vels_addition.append([Z['diff_vel_ch_A2'][i], Z['diff_vel_ch_A3'][i],
                          Z['diff_vel_rep_A2'][i], Z['diff_vel_rep_A3'][i]])

    data = open(os.path.normpath('/'.join([data_dir, 'DigitalPianosDifferencesAddition2.pickle'])), 'rb')
    Z = pickle.load(data)
    digital_piano_d_addition, digital_features_d_addition, digital_vels_addition = [], [], []
    for i in range(len(Z['piano'])):
        digital_piano_d_addition.append(Z['piano'][i])
        digital_features_d_addition.append([Z['diff_cA2'][i][0], Z['diff_cA2'][i][1], Z['diff_cA2'][i][2],
                                   Z['diff_cA3'][i][0], Z['diff_cA3'][i][1], Z['diff_cA3'][i][2],
                                   Z['diff_rA2'][i][0], Z['diff_rA2'][i][1], Z['diff_rA2'][i][2],
                                   Z['diff_rA3'][i][0], Z['diff_rA3'][i][1], Z['diff_rA3'][i][2]])

        digital_vels_addition.append([Z['diff_vel_ch_A2'][i], Z['diff_vel_ch_A3'][i],
                             Z['diff_vel_rep_A2'][i], Z['diff_vel_rep_A3'][i]])

    real_features_d = np.array(real_features_d)
    disk_features_d = np.array(disk_features_d)
    digital_features_d = np.array(digital_features_d)
    disk_piano_d_addition = np.array(disk_piano_d_addition)
    digital_piano_d_addition = np.array(digital_piano_d_addition)
    labels = np.concatenate((real_piano_d, disk_piano_d, digital_piano_d, disk_piano_d_addition, digital_piano_d_addition))
    features = np.concatenate((real_features_d, disk_features_d, digital_features_d, disk_features_d_addition, digital_features_d_addition))
    vels = np.concatenate((real_vels, disk_vels, digital_vels, disk_vels_addition, digital_vels_addition))
    DiffencesFeatures = {'labels': labels, 'features': features, 'vels': vels}

    file_data = open(os.path.normpath('/'.join([data_dir, 'DiffencesFeatures_aligned.pickle'])), 'wb')
    pickle.dump(DiffencesFeatures, file_data)
    file_data.close()
    return

if __name__ == '__main__':

    preparing_features_diff()
    #preparing_features()
