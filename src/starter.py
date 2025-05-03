from PreparingSingleNotes01 import preparing_features, preparing_features_diff
from Organise_features02 import createDataset, createDataset_diff
from Analysis03 import analyze
import argparse
import os
import pickle

"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Start features extraction and analysis.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')
    parser.add_argument('--type', default=['single', 'chord', 'rep'], type=str, nargs='?', help='If consider single notes (single) or chord (chord) and re-triggered notes (rep).')

    return parser.parse_args()


def start_train(args):
    print("######### Preparing for analysis #########")
    print("\n")

    if 'single' in args.type:
        print("######### Single notes #########")
        print("\n")
        preparing_features(data_dir=args.data_dir)
        data = open(os.path.normpath('/'.join([args.data_dir, 'AllFeatures_aligned.pickle'])), 'rb')
        Z = pickle.load(data)
        createDataset(Z, size='win')

    if 'chord' or 'rep' in args.type:
        print("######### Chords and re-triggered notes #########")
        print("\n")
        preparing_features_diff(data_dir=args.data_dir)

        data = open(os.path.normpath('/'.join([args.data_dir, 'DiffencesFeatures_aligned.pickle'])), 'rb')
        Z = pickle.load(data)
        createDataset_diff(Z)

        for type in args.type:
            analyze(type)

def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()