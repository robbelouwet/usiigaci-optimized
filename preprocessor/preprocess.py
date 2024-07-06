import json as js
import os
from segments import SegmentsDataset
from segments.utils import export_dataset
import pycocotools.mask as mask_util
from imantics import Mask
from PIL import Image
import shutil
import sys

# python can't find the file detect.py, this line fixes it, cleaner solutions are more than welcome
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from maskrcnn.prepareDataSet import prepare_data_set


def check_files(files):
    # Check files
    for file in files:
        if not os.path.isfile(file):
            print('File: {} cannot be found'.format(file))
            exit(1)


def load_json(file_name):
    coco_json = os.path.splitext(os.path.basename(file_name))[0] + "_coco.json"
    with open(coco_json, 'r') as file:
        data = file.read()
        json = js.loads(data)

        imgs = json['images']
        anns = json['annotations']
        return imgs, anns


def convert_json(file_names, output_folder_name):
    counter = 0
    json_data = []

    if os.path.isdir('./trainingSet/{}'.format(output_folder_name)):
        shutil.rmtree('./trainingSet/{}'.format(output_folder_name))

    os.mkdir('./trainingSet/{}'.format(output_folder_name))
    os.mkdir('./trainingSet/{}/raw'.format(output_folder_name))

    for file_name in file_names:
        counter += 1

        try:
            dataset = SegmentsDataset(file_name, task='segmentation')

            # Export to COCO format
            export_dataset(dataset, 'coco')

            print("Converting json {}".format(counter))
            images, annotations = load_json(file_name)

            for image in images:
                width = image.get('width')
                height = image.get('height')
                annotations_of_image = []
                regions = []

                # Get all annotations that belong to the current image
                for annotation in annotations:
                    if annotation.get('image_id') == image.get('id'):
                        annotations_of_image.append(annotation)

                # Get the segmentation parts of the annotations
                segments = [item.get('segmentation') for item in annotations_of_image]

                attributes = []
                for segment in segments:
                    list_x = []
                    list_y = []

                    # Decode the coordinates
                    # Looks for the 'counts' variable in the segment to decode from
                    mask = mask_util.decode(segment)[:, :]
                    polygons = Mask(mask).polygons()

                    for element in polygons.points[0]:
                        list_x.append(int(element[0]))
                        list_y.append(int(element[1]))

                    # Write the json which we will append to the file later
                    shape_attributes = {
                        "name": "polygon",
                        "all_points_x": list_x,
                        "all_points_y": list_y
                    }

                    region_attributes = {
                        "name": "crystal"
                    }

                    attributes.append({
                        "shape_attributes": shape_attributes,
                        "region_attributes": region_attributes
                    })

                for attribute in attributes:
                    regions.append(attribute)

                json_data.append({
                    "filename": 'BATCH_{}_{}'.format(counter, image.get('file_name')).replace(' ', '_'),
                    "size": "",
                    "regions": regions,
                    "width": width,
                    "height": height,
                    "file_attributes": {}
                })

            # Get the json which will be used to get the subfolder names in the segments folder
            with open(file_name, 'r') as file:
                json = file.read()
                json = js.loads(json)

            first_folder_name = json['dataset']['owner'] + "_" + json['dataset']['name']

            sub_folder_name = json['name']

            # image_list = os.listdir('./segments/' + str(first_folder_name) + '/' + str(sub_folder_name))
            dir_name = './segments/{}/{}'.format(first_folder_name, sub_folder_name)

            # Move the original images to a separate folder to give as input for Usiigaci. This will also solve an mtAc: CRC error.
            for img_name in os.listdir(dir_name):
                if not img_name.__contains__('label'):
                    img = Image.open(os.path.join(dir_name, img_name))
                    img.save(os.path.join('./trainingSet/{}/raw'.format(output_folder_name),
                                          'BATCH_{}_{}'.format(counter, img_name)).replace(' ', '_'))

        except Exception as e:
            print(e)

        finally:
            # Clean up

            # Remove the coco json
            for file in os.listdir('.'):
                if os.path.splitext(file)[1] == '.json' and os.path.isfile(file):
                    os.remove(file)

            if os.path.exists('./segments'):
                shutil.rmtree('./segments')

    print('Writing json...')
    with open('./trainingSet/{}/masks.json'.format(output_folder_name), 'w+') as outfile:
        js.dump(json_data, outfile)

    # Call the preparedataset
    print('Preparing dataset...')
    prepare_data_set('./trainingSet/{}'.format(output_folder_name))


if __name__ == "__main__":
    parameters = sys.argv[1:]

    if parameters[0].lower() == "--help":
        print("First parameter is the original json to convert.")
        print(
            "Second parameter is optional: the name of the resulting directory that will be created. Default is clear")
        exit(0)
    else:
        file_names = parameters

        result_file_name = "clear"

        if not os.path.isfile(file_names[-1]):
            # Last param is the output-folder
            result_file_name = file_names[-1]
            file_names = file_names[:-1]
            check_files(file_names)
        else:
            output_folder = input('Name of the folder (default is clear): ')
            if len(output_folder) > 0:
                result_file_name = output_folder
            check_files(file_names)

        print("Files will be stored at: ./trainingSet/{}".format(result_file_name))
        convert_json(file_names, output_folder_name=result_file_name)
