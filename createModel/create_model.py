import h5py
import sys

if __name__ == '__main__':
    if sys.argv[1].lower() == '--help':
        print('First argument is the path to the new model, e.g. ./trainingModels/new_model.h5')
        exit(0)

    assert len(sys.argv) == 2, 'Missing arguments. Expecting 1: name of the new h5 model'

    try:
        with h5py.File('{}'.format(sys.argv[1]), 'w'):
            pass
    except Exception as e:
        print('Expected ./trainingModels/{name}.h5 as first parameter.')

    exit(0)
