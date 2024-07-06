import sys
import os
import h5py


def preprocess(files, output_folder_name):
    from preprocessor.preprocess import convert_json

    convert_json(files, output_folder_name=output_folder_name)

    print('Preprocessing: \N{check mark}')


def prepare(dir):
    from maskrcnn.prepareDataSet import prepare_data_set

    # PREPARE
    prepare_data_set(dir)
    print('Preparation of the dataset: \N{check mark}')


def create_weights_model(model_path):
    with h5py.File(model_path, 'w'):
        pass
    return os.path.exists(model_path)


def check_weights_model(created):
    if created:
        print('Created weights model.')
    else:
        print('Could not create weights model.')
        exit(1)


def train(dir, model_path, use_new_model=False):
    import maskrcnn.train as training

    print('Started training')
    try:
        # Calling train() in train.py does not print the elapsed time as well after training.
        import timeit
        start_time = timeit.default_timer()

        training.train_by_path(dir, model_path, use_new_model=use_new_model)

        elapsed_time = timeit.default_timer()
        elapsed_time = elapsed_time - start_time
        hours = elapsed_time / (60 * 60)
        print(f"Elapsed time: {elapsed_time}. Hours: {hours}")

    except IndexError as ie:
        print(ie)
        print(
            'An indexerror has occurred. This may result from an invalid model buildup. '
            'Please check that the given model has the right properties such as size (~'
            '255MB). If this issue persists, create a new model.')
        exit(1)
    print('Training: \N{check mark}')


def detect(type_parameter, input_parameter, model_path):
    import maskrcnn.detect_tester as detecter

    print('Started detection')
    detecter.main(type_parameter, input_parameter, model_path)
    print('Detection: \N{check mark}')


def train_prerequisites(model_path):
    if not os.path.exists(model_path):
        print('No model has been found at {}. A new one will be made'.format(model_path))
        created = create_weights_model(model_path)
        check_weights_model(created)

        # We are creating a new model that will have to be made in train.py @ line 299
        return True

    else:
        new_model = ""
        while new_model not in ['new', 'improve']:
            new_model = input('A model already exists at {}. Do you want to start a new model or improve the existing '
                              'one? '
                              '(new/improve): '.format(model_path))

        if new_model.lower() == 'new':
            confirmation = ""
            while confirmation not in ['y', 'n', 'yes', 'no']:
                confirmation = input('Are you sure? The current model will be deleted! (y/n): ')
                if confirmation.lower() not in ['y', 'n', 'yes', 'no']:
                    print('Accepted answers: yes/y/no/n')

            if confirmation.lower() in ['yes', 'y']:
                if os.path.exists(model_path):
                    os.remove(model_path)
                created = create_weights_model(model_path)
                check_weights_model(created)
                print('New model created at: {}'.format(model_path))

            # We are creating a new model that will have to be made in train.py @ line 299
            return True
        else:
            return False


def help():
    print('This program will handle all maskrcnn flows.')
    print('Possible actions: prepare | train | detect')
    print('Run as: \n')
    print('python run.py {action} {target_input}')
    print('If you want to detect: run.py detect image/folder target_image/target_folder')


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        action = sys.argv[1].lower()

        # The input target will probably be the folder we want to use for training,
        # but can also be a json file or detection image
        input_target = sys.argv[2]

        if action not in ['prepare', 'train', 'detect']:
            help()
            exit(1)

        if action == 'prepare':
            # PREPARE

            if not os.path.exists(sys.argv[2]):
                print('Could not find input target')
                exit(1)

            output_file_name = 'clear'

            _, file_extension = os.path.splitext(sys.argv[2])
            if file_extension == '.json':
                # JSON -> preprocess
                if not os.path.splitext(sys.argv[-1])[1].replace(' ', '') == '.json':
                    input_targets = sys.argv[2:-1]
                    output_file_name = sys.argv[-1]
                else:
                    input_targets = sys.argv[2:]

                try:
                    preprocess(input_targets, output_file_name)
                except Exception as e:
                    print('Could not preprocess json')
                    print(e)
                    exit(1)
            else:
                # PREPARE

                try:
                    prepare(input_target)
                except Exception as e:
                    print('Could not prepare dataset')
                    print(e)
                    exit(1)

            answer = ""
            while answer not in ['yes', 'y', 'no', 'n']:
                answer = input('Do you want to train this new dataset? (y/n): ')

            if answer in ['yes', 'y']:
                model_path = 'model.h5'

                different_model = ""
                while different_model not in ['yes', 'y', 'no', 'n']:
                    different_model = input('You want to use the default model at ./trainingModels/model.h5? (y/n): ')

                if different_model in ['no', 'n']:
                    model_path = input('Please enter the name of the new model: ./trainingModels/')

                model_path = os.path.join('./trainingModels', model_path)
                if not os.path.isfile(model_path):
                    print('Could not find the model file')
                    exit(1)

                use_new_model = train_prerequisites(model_path)
                train(os.path.join('./trainingSet', output_file_name), model_path, use_new_model=use_new_model)

        if action == 'train':
            # TRAIN

            if not os.path.exists(input_target):
                print('Could not find input target')
                exit(1)

            model_path = './trainingModels/model.h5'

            if len(sys.argv) == 4:
                model_path = sys.argv[3]

            use_new_model = train_prerequisites(model_path)

            train(input_target, model_path, use_new_model=use_new_model)

        if action == 'detect':
            # DETECT

            if not len(sys.argv) >= 4:
                print('No input image or directory given.')
                print('Expected: run.py detect image/folder path/to/imageORfolder')
                exit(1)

            if not os.path.exists(sys.argv[3]):
                print('Could not find input target')
                exit(1)

            else:
                model_path = "./trainingModels/model.h5"

                if len(sys.argv) == 5:
                    model_path = sys.argv[4]

                type_parameter = input_target
                input_target = sys.argv[3]
                detect(type_parameter, input_target, model_path)

                # Create output paths to indicate where the files are stored
                img_name_no_extension = os.path.basename(input_target).split('.')[0]

                if type_parameter == 'image':
                    output_base_path = os.path.join(os.path.dirname(input_target), 'out')
                    output_image_path = os.path.join(output_base_path, img_name_no_extension)
                else:
                    output_image_path = os.path.join(input_target, 'out')

                print('Saved your detection at {}'.format(output_image_path))
    else:
        help()
        exit(1)
