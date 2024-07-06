import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from maskrcnn.evaluate_predictions import evaluate, print_evaluation_metrics
from project_utils import ROOT_DIR
import maskrcnn.detect as detect


def main(type_parameter, path_parameter, model_path):
    if type_parameter != 'image' and type_parameter != 'folder' and type_parameter != 'folders':
        print('Invalid arguments were given')
        print('First argument must be \'image\' or \'folder\' or \'folders\'')
        exit(1)

    detector = detect.Detect(model_path)

    if type_parameter == 'image':
        extension = os.path.splitext(path_parameter)[1]
        if extension == ".png" or extension == ".bmp":
            detector.detectSingle(path_parameter, 'general', 0.87)
        else:
            print("Image can't be of type %s" % extension)
            exit(1)
    if type_parameter == 'folder':
        if os.path.isdir(path_parameter):
            detector.detectMultiple(path_parameter, 'general', 0.87)
        else:
            print("%s is not a folder" % path_parameter)
            exit(1)

    if type_parameter == 'folders':
        for i in range(len(path_parameter)):
            print(i)
            if os.path.exists(path_parameter[i]):
                if len(os.listdir(path_parameter[i])) > 0:
                    detector = detect.Detect(model_path)
                    detector.detectMultiple(path_parameter[i], 'general', 0.87)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Missing parameters.')
        exit(1)
    evaluate('/'.join(sys.argv[2].split('/')[:-2]))

    typeParameter = sys.argv[1]
    pathParameter = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else './trainingModels/model.h5'
    main(typeParameter, pathParameter, model_path)
