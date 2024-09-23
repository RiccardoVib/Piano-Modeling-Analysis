import numpy as np
import os
import pickle
from FeatureExtraction import Spectral_Kurtosis, Spectral_Skewness

def createDataset(Z, size):

    
    """
      create the feature vectors for the single notes case, which include the audio descriptors differences of different velocities
    """

    
    real_piano = Z['real_piano']
    real_features = Z['real_features']
    disk_piano = Z['disk_piano']
    disk_features = Z['disk_features']
    digital_piano = Z['digital_piano']
    digital_features = Z['digital_features']
    del Z
    arr = [-1, 6, 14, 19, 30, 35, 44, 50, 57, 65, 78, 93, 104, 108, 115, 126, 131, 138, 150, 157, 163] # indexes to pick the notes at specific chosen velocities
    p, f = [], []
    for i in range(len(arr) - 1):
        p.append(real_piano[arr[i] + 1:arr[i + 1]])
        f.append(real_features[arr[i] + 1:arr[i + 1]])

    examples = {'pianos': [], 'labels': [], 'note': [], 'f': [], 'c': [], 'd': [], 'type': []}
    for i in range(len(p)): #number of notes
        for h in range(len(p[i])-1): #n of vels
            #distance = np.mean([p[i][h+1][2], p[i][h][2]])
            distance = [p[i][h+1][2], p[i][h][2]]
            if distance != 0:
                temp = np.array(f[i][h+1][0]).reshape(-1)
                sub1 = temp
                temp = np.array(f[i][h][0]).reshape(-1)
                sub2 = temp
                for n in range(1, len(f[i][h])):
                    temp = np.array(f[i][h+1][n]).reshape(-1)
                    sub1 = np.concatenate((sub1, temp))
                    temp = np.array(f[i][h][n]).reshape(-1)
                    sub2 = np.concatenate((sub2, temp))
                diff = sub1 - sub2#(np.array(f[i][h+1]) - np.array(f[i][h]))
                #diff = diff / distance
                examples['pianos'].append(p[i][h][0])
                examples['note'].append(p[i][h][1])
                examples['labels'].append(0)
                examples['f'].append(diff)
                examples['d'].append(distance)
    for i in range(len(disk_piano)-1):  # number of notes
        if disk_piano[i][1] == disk_piano[i+1][1]: #n of vels
            #distance = np.mean([disk_piano[i+1][2], disk_piano[i][2]])
            distance = [disk_piano[i+1][2], disk_piano[i][2]]
            temp = np.array(disk_features[i + 1][0]).reshape(-1)
            sub1 = temp
            temp = np.array(disk_features[i][0]).reshape(-1)
            sub2 = temp
            for n in range(1, len(disk_features[i])):
                temp = np.array(disk_features[i + 1][n]).reshape(-1)
                sub1 = np.concatenate((sub1, temp))
                temp = np.array(disk_features[i][n]).reshape(-1)
                sub2 = np.concatenate((sub2, temp))

            diff = sub1 - sub2
            #diff = diff / distance
            examples['pianos'].append(disk_piano[i][0])
            examples['labels'].append(0)
            examples['note'].append(disk_piano[i][1])
            examples['f'].append(diff)
            examples['d'].append(distance)
        else:
            continue

    for i in range(len(digital_piano[:204])-1):  # number of notes
        if digital_piano[i][1] == digital_piano[i+1][1] and digital_piano[i][0] == digital_piano[i+1][0]: #n of vels
            #distance = np.mean([digital_piano[i+1][2], digital_piano[i][2]])#(digital_piano[i+1][2] - digital_piano[i][2])
            distance = [digital_piano[i+1][2], digital_piano[i][2]]
            temp = np.array(digital_features[i + 1][0]).reshape(-1)
            sub1 = temp
            temp = np.array(digital_features[i][0]).reshape(-1)
            sub2 = temp
            for n in range(1, len(digital_features[i])):
                temp = np.array(digital_features[i + 1][n]).reshape(-1)
                sub1 = np.concatenate((sub1, temp))
                temp = np.array(digital_features[i][n]).reshape(-1)
                sub2 = np.concatenate((sub2, temp))

            diff = (sub1 - sub2)
            #diff = diff / distance
            examples['pianos'].append(digital_piano[i][0])
            examples['labels'].append(1)
            examples['note'].append(digital_piano[i][1])
            examples['f'].append(diff)
            examples['d'].append(distance)
        else:
            continue

    for i in range(len(digital_piano[204:])-1):  # number of notes
        if digital_piano[204+i][1] == digital_piano[204+i+1][1] and digital_piano[204+i][0] == digital_piano[204+i+1][0]: #n of vels
            #distance = np.mean([digital_piano[i+1][2], digital_piano[i][2]])#(digital_piano[i+1][2] - digital_piano[i][2])
            distance = [digital_piano[204+i+1][2], digital_piano[204+i][2]]
            temp = np.array(digital_features[204+i + 1][0]).reshape(-1)
            sub1 = temp
            temp = np.array(digital_features[204+i][0]).reshape(-1)
            sub2 = temp
            for n in range(1, len(digital_features[204+i])):
                temp = np.array(digital_features[204+i + 1][n]).reshape(-1)
                sub1 = np.concatenate((sub1, temp))
                temp = np.array(digital_features[204+i][n]).reshape(-1)
                sub2 = np.concatenate((sub2, temp))
            diff = (sub1 - sub2)
            #diff = diff / distance
            examples['pianos'].append(digital_piano[204+i][0])
            examples['labels'].append(2)#disk_piano[i][0])
            examples['note'].append(digital_piano[i][1])
            examples['f'].append(diff)
            examples['d'].append(distance)
        else:
            continue
   
    file_data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_aligned.pickle'])), 'wb')

    c = []
    type_of_piano = []
    for i in range(len(examples['pianos'])):
        if examples['pianos'][i] == 'Schimmel' or examples['pianos'][i] == 'YamahaMidi' or examples['pianos'][i] == 'Disk2':
            if examples['note'][i] == 'A3Rep' or examples['note'][i] == 'A2Rep' or examples['note'][i] == 'A3Chord' or examples['note'][i] == 'A2Chord':
                c.append('blue')
            else:
                c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples['pianos'][i] == 'Kont1' or examples['pianos'][i] == 'Kont2' or examples['pianos'][i] == 'Kont3':
            if examples['note'][i] == 'A3Rep' or examples['note'][i] == 'A2Rep' or examples['note'][i] == 'A3Chord' or examples['note'][i] == 'A2Chord':
                c.append('red')
            else:
                c.append('red')
            type_of_piano.append('Sample-based')

        elif examples['pianos'][i] == 'Teq1' or examples['pianos'][i] == 'Teq2' or examples['pianos'][i] == 'Teq3':
            if examples['note'][i] == 'A3Rep' or examples['note'][i] == 'A2Rep' or examples['note'][i] == 'A3Chord' or examples['note'][i] == 'A2Chord':
                c.append('green')
            else:
                c.append('green')
            type_of_piano.append('Physic-based')

    c = np.array(c)
    examples['c'].append(c)
    examples['type'].append(type_of_piano)
    pickle.dump(examples, file_data)
    file_data.close()
    return
##################

def createDataset_diff(Z):
    """
      create the feature vectors for chords and repeated notes cases, which include the audio descriptors differences of different velocities
    """

    examples_ch = {'pianos': [], 'labels': [], 'f': [], 'vels': [], 'c': [], 'type': []}
    examples_rep = {'pianos': [], 'labels': [], 'f': [], 'vels': [], 'c': [], 'type': []}

    features = np.array(Z['features'])
    labels = np.array(Z['labels'])
    vels = np.array(Z['vels'])

    for i in range(len(labels)):# number of notes
        piano = features[i]
        v = np.array(vels[i]).reshape(-1)

        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])

        if labels[i] == 'Schimmel' or labels[i] == 'YamahMidi' or labels[i] == 'Disk2':
            lab = 0
        elif labels[i] == 'Kont1' or labels[i] == 'Kont2' or labels[i] == 'Kont3':
            lab = 1
        elif labels[i] == 'Teq1' or labels[i] == 'Teq2' or labels[i] == 'Teq3':
            lab = 2

        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)


        #diff_cA2, diff_cA3, diff_rA2, diff_rA3
        examples_ch['f'].append([np.mean(piano[0, 0]), np.var(piano[0, 0]), np.abs(Spectral_Skewness(piano[0, 0])), np.abs(Spectral_Kurtosis(piano[0, 0])),
                                 np.mean(piano[0, 1]), np.var(piano[0, 1]), np.abs(Spectral_Skewness(piano[0, 1])), np.abs(Spectral_Kurtosis(piano[0, 1])),
                                 np.mean(piano[0, 2]), np.var(piano[0, 2]), np.abs(Spectral_Skewness(piano[0, 2])), np.abs(Spectral_Kurtosis(piano[0, 2])),
                                 np.mean(piano[0, 3]), np.var(piano[0, 3]), np.abs(Spectral_Skewness(piano[0, 3])), np.abs(Spectral_Kurtosis(piano[0, 3])),
                                 np.mean(piano[0, 4]), np.var(piano[0, 4]), np.abs(Spectral_Skewness(piano[0, 4])), np.abs(Spectral_Kurtosis(piano[0, 4])),
                                 v[0]])
        examples_ch['f'].append([np.mean(piano[1, 0]), np.var(piano[1, 0]), np.abs(Spectral_Skewness(piano[1, 0])), np.abs(Spectral_Kurtosis(piano[1, 0])),
                                 np.mean(piano[1, 1]), np.var(piano[1, 1]), np.abs(Spectral_Skewness(piano[1, 1])), np.abs(Spectral_Kurtosis(piano[1, 1])),
                                 np.mean(piano[1, 2]), np.var(piano[1, 2]), np.abs(Spectral_Skewness(piano[1, 2])), np.abs(Spectral_Kurtosis(piano[1, 2])),
                                 np.mean(piano[1, 3]), np.var(piano[1, 3]), np.abs(Spectral_Skewness(piano[1, 3])), np.abs(Spectral_Kurtosis(piano[1, 3])),
                                 np.mean(piano[1, 4]), np.var(piano[1, 4]), np.abs(Spectral_Skewness(piano[1, 4])), np.abs(Spectral_Kurtosis(piano[1, 4])),
                                 v[1]])
        examples_ch['f'].append([np.mean(piano[2, 0]), np.var(piano[2, 0]), np.abs(Spectral_Skewness(piano[2, 0])), np.abs(Spectral_Kurtosis(piano[2, 0])),
                                 np.mean(piano[2, 1]), np.var(piano[2, 1]), np.abs(Spectral_Skewness(piano[2, 1])), np.abs(Spectral_Kurtosis(piano[2, 1])),
                                 np.mean(piano[2, 2]), np.var(piano[2, 2]), np.abs(Spectral_Skewness(piano[2, 2])), np.abs(Spectral_Kurtosis(piano[2, 2])),
                                 np.mean(piano[2, 3]), np.var(piano[2, 3]), np.abs(Spectral_Skewness(piano[2, 3])), np.abs(Spectral_Kurtosis(piano[2, 3])),
                                 np.mean(piano[2, 4]), np.var(piano[2, 4]), np.abs(Spectral_Skewness(piano[2, 4])), np.abs(Spectral_Kurtosis(piano[2, 4])),
                                 v[2]])
        examples_ch['f'].append([np.mean(piano[3, 0]), np.var(piano[3, 0]), np.abs(Spectral_Skewness(piano[3, 0])), np.abs(Spectral_Kurtosis(piano[3, 0])),
                                 np.mean(piano[3, 1]), np.var(piano[3, 1]), np.abs(Spectral_Skewness(piano[3, 1])),
                                 np.abs(Spectral_Kurtosis(piano[3, 1])),
                                 np.mean(piano[3, 2]), np.var(piano[3, 2]), np.abs(Spectral_Skewness(piano[3, 2])),
                                 np.abs(Spectral_Kurtosis(piano[0, 2])),
                                 np.mean(piano[3, 3]), np.var(piano[3, 3]), np.abs(Spectral_Skewness(piano[3, 3])),
                                 np.abs(Spectral_Kurtosis(piano[0, 3])),
                                 np.mean(piano[3, 4]), np.var(piano[3, 4]), np.abs(Spectral_Skewness(piano[3, 4])),
                                 np.abs(Spectral_Kurtosis(piano[3, 4])),
                                 v[3]])
        examples_ch['f'].append([np.mean(piano[4, 0]), np.var(piano[4, 0]), np.abs(Spectral_Skewness(piano[4, 0])), np.abs(Spectral_Kurtosis(piano[4, 0])),
                                 np.mean(piano[4, 1]), np.var(piano[4, 1]), np.abs(Spectral_Skewness(piano[4, 1])),
                                 np.abs(Spectral_Kurtosis(piano[4, 1])),
                                 np.mean(piano[4, 2]), np.var(piano[4, 2]), np.abs(Spectral_Skewness(piano[4, 2])),
                                 np.abs(Spectral_Kurtosis(piano[4, 2])),
                                 np.mean(piano[4, 3]), np.var(piano[4, 3]), np.abs(Spectral_Skewness(piano[4, 3])),
                                 np.abs(Spectral_Kurtosis(piano[4, 3])),
                                 np.mean(piano[4, 4]), np.var(piano[4, 4]), np.abs(Spectral_Skewness(piano[4, 4])),
                                 np.abs(Spectral_Kurtosis(piano[4, 4])),
                                 v[4]])
        examples_ch['f'].append([np.mean(piano[5, 0]), np.var(piano[5, 0]), np.abs(Spectral_Skewness(piano[5, 0])), np.abs(Spectral_Kurtosis(piano[5, 0])),
                                 np.mean(piano[5, 1]), np.var(piano[5, 1]), np.abs(Spectral_Skewness(piano[5, 1])),
                                 np.abs(Spectral_Kurtosis(piano[5, 1])),
                                 np.mean(piano[5, 2]), np.var(piano[5, 2]), np.abs(Spectral_Skewness(piano[5, 2])),
                                 np.abs(Spectral_Kurtosis(piano[5, 2])),
                                 np.mean(piano[5, 3]), np.var(piano[5, 3]), np.abs(Spectral_Skewness(piano[5, 3])),
                                 np.abs(Spectral_Kurtosis(piano[5, 3])),
                                 np.mean(piano[5, 4]), np.var(piano[5, 4]), np.abs(Spectral_Skewness(piano[5, 4])),
                                 np.abs(Spectral_Kurtosis(piano[5, 4])),
                                 v[5]])

        examples_rep['f'].append([np.mean(piano[6, 0]), np.var(piano[6, 0]), np.abs(Spectral_Skewness(piano[6, 0])), np.abs(Spectral_Kurtosis(piano[6, 0])),
                                  np.mean(piano[6, 1]), np.var(piano[6, 1]), np.abs(Spectral_Skewness(piano[6, 1])),
                                  np.abs(Spectral_Kurtosis(piano[6, 1])),
                                  np.mean(piano[6, 2]), np.var(piano[6, 2]), np.abs(Spectral_Skewness(piano[6, 2])),
                                  np.abs(Spectral_Kurtosis(piano[6, 2])),
                                  np.mean(piano[6, 3]), np.var(piano[6, 3]), np.abs(Spectral_Skewness(piano[6, 3])),
                                  np.abs(Spectral_Kurtosis(piano[6, 3])),
                                  np.mean(piano[6, 4]), np.var(piano[6, 4]), np.abs(Spectral_Skewness(piano[6, 4])),
                                  np.abs(Spectral_Kurtosis(piano[6, 4])),
                                  v[6]])
        examples_rep['f'].append([np.mean(piano[7, 0]), np.var(piano[7, 0]), np.abs(Spectral_Skewness(piano[7, 0])), np.abs(Spectral_Kurtosis(piano[7, 0])),
                                  np.mean(piano[7, 1]), np.var(piano[7, 1]), np.abs(Spectral_Skewness(piano[7, 1])),
                                  np.abs(Spectral_Kurtosis(piano[7, 1])),
                                  np.mean(piano[7, 2]), np.var(piano[7, 2]), np.abs(Spectral_Skewness(piano[7, 2])),
                                  np.abs(Spectral_Kurtosis(piano[7, 2])),
                                  np.mean(piano[7, 3]), np.var(piano[7, 3]), np.abs(Spectral_Skewness(piano[7, 3])),
                                  np.abs(Spectral_Kurtosis(piano[7, 3])),
                                  np.mean(piano[7, 4]), np.var(piano[7, 4]), np.abs(Spectral_Skewness(piano[7, 4])),
                                  np.abs(Spectral_Kurtosis(piano[7, 4])),
                                  v[7]])
        examples_rep['f'].append([np.mean(piano[8, 0]), np.var(piano[8, 0]), np.abs(Spectral_Skewness(piano[8, 0])), np.abs(Spectral_Kurtosis(piano[8, 0])),
                                  np.mean(piano[8, 1]), np.var(piano[8, 1]), np.abs(Spectral_Skewness(piano[8, 1])),
                                  np.abs(Spectral_Kurtosis(piano[8, 1])),
                                  np.mean(piano[8, 2]), np.var(piano[8, 2]), np.abs(Spectral_Skewness(piano[8, 2])),
                                  np.abs(Spectral_Kurtosis(piano[8, 2])),
                                  np.mean(piano[8, 3]), np.var(piano[8, 3]), np.abs(Spectral_Skewness(piano[8, 3])),
                                  np.abs(Spectral_Kurtosis(piano[8, 3])),
                                  np.mean(piano[8, 4]), np.var(piano[8, 4]), np.abs(Spectral_Skewness(piano[8, 4])),
                                  np.abs(Spectral_Kurtosis(piano[8, 4])),
                                  v[8]])
        examples_rep['f'].append([np.mean(piano[9, 0]), np.var(piano[9, 0]), np.abs(Spectral_Skewness(piano[9, 0])), np.abs(Spectral_Kurtosis(piano[9, 0])),
                                  np.mean(piano[9, 1]), np.var(piano[9, 1]), np.abs(Spectral_Skewness(piano[9, 1])),
                                  np.abs(Spectral_Kurtosis(piano[9, 1])),
                                  np.mean(piano[9, 2]), np.var(piano[9, 2]), np.abs(Spectral_Skewness(piano[9, 2])),
                                  np.abs(Spectral_Kurtosis(piano[9, 2])),
                                  np.mean(piano[9, 3]), np.var(piano[9, 3]), np.abs(Spectral_Skewness(piano[9, 3])),
                                  np.abs(Spectral_Kurtosis(piano[0, 3])),
                                  np.mean(piano[9, 4]), np.var(piano[9, 4]), np.abs(Spectral_Skewness(piano[9, 4])),
                                  np.abs(Spectral_Kurtosis(piano[9, 4])),
                                  v[9]])
        examples_rep['f'].append([np.mean(piano[10, 0]), np.var(piano[10, 0]), np.abs(Spectral_Skewness(piano[10, 0])), np.abs(Spectral_Kurtosis(piano[10, 0])),
                                  np.mean(piano[10, 1]), np.var(piano[10, 1]), np.abs(Spectral_Skewness(piano[10, 1])),
                                  np.abs(Spectral_Kurtosis(piano[10, 1])),
                                  np.mean(piano[10, 2]), np.var(piano[10, 2]), np.abs(Spectral_Skewness(piano[10, 2])),
                                  np.abs(Spectral_Kurtosis(piano[10, 2])),
                                  np.mean(piano[10, 3]), np.var(piano[10, 3]), np.abs(Spectral_Skewness(piano[10, 3])),
                                  np.abs(Spectral_Kurtosis(piano[10, 3])),
                                  np.mean(piano[10, 4]), np.var(piano[10, 4]), np.abs(Spectral_Skewness(piano[10, 4])),
                                  np.abs(Spectral_Kurtosis(piano[10, 4])),
                                  v[10]])
        examples_rep['f'].append([np.mean(piano[11, 0]), np.var(piano[11, 0]), np.abs(Spectral_Skewness(piano[11, 0])), np.abs(Spectral_Kurtosis(piano[11, 0])),
                                  np.mean(piano[11, 1]), np.var(piano[11, 1]), np.abs(Spectral_Skewness(piano[11, 1])),
                                  np.abs(Spectral_Kurtosis(piano[11, 1])),
                                  np.mean(piano[11, 2]), np.var(piano[11, 2]), np.abs(Spectral_Skewness(piano[11, 2])),
                                  np.abs(Spectral_Kurtosis(piano[11, 2])),
                                  np.mean(piano[11, 3]), np.var(piano[11, 3]), np.abs(Spectral_Skewness(piano[11, 3])),
                                  np.abs(Spectral_Kurtosis(piano[11, 3])),
                                  np.mean(piano[11, 4]), np.var(piano[11, 4]), np.abs(Spectral_Skewness(piano[11, 4])),
                                  np.abs(Spectral_Kurtosis(piano[11, 4])),
                                  v[11]])

    file_data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_diff.pickle'])), 'wb')

    c = []
    type_of_piano = []

    for i in range(len(examples_ch['pianos'])):
        if examples_ch['pianos'][i] == 'Schimmel':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'YamahaMidi':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'Disk2':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'Kont1':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Kont2':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Kont3':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Teq1':
            c.append('green')
            type_of_piano.append('Physic-based')
        elif examples_ch['pianos'][i] == 'Teq2':
            c.append('green')
            type_of_piano.append('Physic-based')
        elif examples_ch['pianos'][i] == 'Teq3':
            c.append('green')
            type_of_piano.append('Physic-based')

    c = np.array(c)
    examples_ch['c'].append(c)
    examples_rep['c'].append(c)
    examples_ch['type'].append(type_of_piano)
    examples_rep['type'].append(type_of_piano)

    examples = {'examples_ch': examples_ch, 'examples_rep': examples_rep}
    pickle.dump(examples, file_data)
    file_data.close()
    return



def createDataset_diff2(Z):

    examples_ch = {'pianos': [], 'labels': [], 'f': [], 'vels': [], 'c': [], 'type': []}
    examples_rep = {'pianos': [], 'labels': [], 'f': [], 'vels': [], 'c': [], 'type': []}

    features = np.array(Z['features'])
    labels = np.array(Z['labels'])
    vels = np.array(Z['vels'])

    for i in range(len(labels)):# number of notes
        piano = features[i]
        v = np.array(vels[i]).reshape(-1)

        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_ch['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])
        examples_rep['pianos'].append(labels[i])

        if labels[i] == 'Schimmel' or labels[i] == 'YamahMidi' or labels[i] == 'Disk2':
            lab = 0
        elif labels[i] == 'Kont1' or labels[i] == 'Kont2' or labels[i] == 'Kont3':
            lab = 1
        elif labels[i] == 'Teq1' or labels[i] == 'Teq2' or labels[i] == 'Teq3':
            lab = 2

        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_ch['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)
        examples_rep['labels'].append(lab)


        #diff_cA2, diff_cA3, diff_rA2, diff_rA3
        examples_ch['f'].append([np.concatenate((piano[0], np.array(v[0]).reshape(-1)))])
        examples_ch['f'].append([np.concatenate((piano[1], np.array(v[1]).reshape(-1)))])
        examples_ch['f'].append([np.concatenate((piano[2], np.array(v[2]).reshape(-1)))])
        examples_ch['f'].append([np.concatenate((piano[3], np.array(v[3]).reshape(-1)))])
        examples_ch['f'].append([np.concatenate((piano[4], np.array(v[4]).reshape(-1)))])
        examples_ch['f'].append([np.concatenate((piano[5], np.array(v[5]).reshape(-1)))])

        examples_rep['f'].append([np.concatenate((piano[6], np.array(v[6]).reshape(-1)))])
        examples_rep['f'].append([np.concatenate((piano[7], np.array(v[7]).reshape(-1)))])
        examples_rep['f'].append([np.concatenate((piano[8], np.array(v[8]).reshape(-1)))])
        examples_rep['f'].append([np.concatenate((piano[9], np.array(v[9]).reshape(-1)))])
        examples_rep['f'].append([np.concatenate((piano[10], np.array(v[10]).reshape(-1)))])
        examples_rep['f'].append([np.concatenate((piano[11], np.array(v[11]).reshape(-1)))])

    file_data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_diff_aligned.pickle'])), 'wb')

    c = []
    type_of_piano = []

    for i in range(len(examples_ch['pianos'])):
        if examples_ch['pianos'][i] == 'Schimmel':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'YamahaMidi':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'Disk2':
            c.append('blue')
            type_of_piano.append('Acoustic')
        elif examples_ch['pianos'][i] == 'Kont1':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Kont2':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Kont3':
            c.append('red')
            type_of_piano.append('Sampled-based')
        elif examples_ch['pianos'][i] == 'Teq1':
            c.append('green')
            type_of_piano.append('Physic-based')
        elif examples_ch['pianos'][i] == 'Teq2':
            c.append('green')
            type_of_piano.append('Physic-based')
        elif examples_ch['pianos'][i] == 'Teq3':
            c.append('green')
            type_of_piano.append('Physic-based')

    c = np.array(c)
    examples_ch['c'].append(c)
    examples_rep['c'].append(c)
    examples_ch['type'].append(type_of_piano)
    examples_rep['type'].append(type_of_piano)

    examples = {'examples_ch': examples_ch, 'examples_rep': examples_rep}
    pickle.dump(examples, file_data)
    file_data.close()
    return

if __name__ == '__main__':

    data_dir = '../../../Analysis/Note_collector'

    #data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_2_smaller.pickle'])), 'rb')
    #data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_2_bigger.pickle'])), 'rb')
    #data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_2.pickle'])), 'rb')
    #data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_2_ultrasmall.pickle'])), 'rb')
    data = open(os.path.normpath('/'.join([data_dir, 'AllFeatures_aligned.pickle'])), 'rb')
    Z = pickle.load(data)
    #createDataset(Z, size='win')

    data = open(os.path.normpath('/'.join([data_dir, 'DiffencesFeatures_all.pickle'])), 'rb')
    Z = pickle.load(data)
    #createDataset_diff(Z)

    data = open(os.path.normpath('/'.join([data_dir, 'DiffencesFeatures_aligned.pickle'])), 'rb')
    Z = pickle.load(data)
    createDataset_diff2(Z)
