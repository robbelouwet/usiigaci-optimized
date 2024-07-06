Usiigaci-Optimized

The cleaned up version of the Crysco-project.

More information about the Usiigaci research can be found @ https://github.com/oist/Usiigaci & https://github.com/matterport/Mask_RCNN

## TIPS

Replace the paths with **your own paths**.

You might've added some python versions to your $PATH system environment variables. You can check this in:

- windows button + s -> path -> edit the system environment variables -> environment variables

## CONFIGURATION

### Virtual environment

It is recommended to create a virtual environment with Python == 3.7 as interpreter before installing the
requirements.txt file. Execute with python3 if using python version 3.x

1. If you're not there already, change your directory to usiigaci-optimized.

```powershell
cd path/to/usiigaci-optimized
```

2. Install virtualenv.

```powershell
pip install virtualenv
```

Execute the following command.

```powershell
python -m venv venv/
```

Check your terminal if your line starts with (venv). If it doesn't, use the following command:

```powershell
Windows : .\venv\Scripts\activate
Mac + Linux: source venv/bin/activate
```

### Installing requirements.txt

Before we can start preparing our data we have to install the dependencies.

```powershell
pip install -r requirements.txt
```

## RUNNING THE SOFTWARE
The software can be used for either .bpm and .png images.

### Dataset filestructure

This is the minimum filestructure that you'll need for your training dataset.

```text
.  
├── ...  
├── trainingSet                    # TrainingSet folder  
│   ├── clear                      # Example of one crystal type  
│   │   ├── raw                    # Folder that contains the raw images  
│   │   └── masks.json             # Json that contains all features of all images 
│   ├── ...
└── ...
```

There are two options:

- Start the training using the run.py script.
- Follow every step detailed further.

##

## Run using the run.py script:

```powershell
Windows + Mac : python run.py {action} {trainingset}
Linux : sudo venv/bin/python run.py {action} {trainingset}
```

### The run will execute the action you pass as the 1st parameter. This can be either: prepare, train or detect.

#### -To PREPARE you can either pass along a pre-made folder corresponding to the
  aforementioned [filestructure](#dataset-filestructure)
  or a json which will be processed and prepared afterwards.

Using the pre-made folder:

```powershell
Windows + Mac : python run.py prepare ./trainingSet/clear
Linux : sudo venv/bin/python run.py prepare ./trainingSet/clear
```

Prepare a dataset from one or more json files:

You can pass more than one json file as argument. All the images will automatically be arranged for you. 
As last parameter you can pass along the name of the output folder in which the files will be stored. Default is 'clear'. 
You need a stable internet connection to perform this step.

```powershell
Windows + Mac : python run.py prepare ./json/5procent.json ./json/10procent.json {folder name}
Linux : sudo venv/bin/python run.py prepare ./json/5procent.json ./json/10procent.json {folder name}
```

#### -To TRAIN:

It's possible that when training on multiple GPU's will result in a warning that mentions that CPU workers die. 
This warning won't affect the training but will slow it down drastically
This warning can be resolved by increasing the time out at: venv/lib/python3.7/site-packages/keras/utils/data_utils.py:610 or removing the
timeout entirely.

- To train you can pass 2 parameters.
    1) The folder that contains the training data. This could be for example: ./trainingSet/clear
    2) The model that will be used. Default is ./trainingModels/model.h5

  When you start the training the session will ask you if you want to either improve the given model or create it as a
  new one. If the given model cannot be found at the specified path, a new one will be created and used to train.

  The training will search for the directories made in the prepare step. If they cannot be found, the preparation will
  automatically start. The training will automatically calculate the optimal number of steps for each epoch.

  Parameters: in train.py you can customize the configuration of the training. Some parameters like steps per epoch and
  batch sizes will by default be auto generated, but it's possible to override these by defining them explicitly.

```powershell
Windows + Mac: python  run.py  train  ./trainingSet/clear
Linux : sudo venv/bin/python  run.py  train  ./trainingSet/clear
```

We advise using a tmux session so that training can run in the background:

To create a terminal:
```powershell
Linux : tmux new -t {name}
```

To leave/detach from a tmux:
```powershell
Linux : ctrl + b -> ':' -> detach
```

To attach to a terminal:
```powershell
Linux : tmux attach  OR  tmux a -t {name}
```

##### Current best trained model
The current best trained model can be found at ./trainingModels/best_model.h5.

#### -To DETECT:

To detect an image or folder you pass the type as 2nd parameter and input as 3rd.

```powershell
Windows + Mac: python run.py detect image ./trainingSet/clear/raw/BATCH_1_IMG_aceton_(1).png
Linux : sudo venv/bin/python run.py detect image ./trainingSet/clear/raw/BATCH_1_IMG_aceton_(1).png
```

You can find the output of the detection in the 'out' folder that will be created at the location you provided as the
source of your training data. For example: if you provide ./trainingSet/clear, the out folder can be found in:
./trainingSet/clear/raw/out/{image_name}

A detection will create a few files:

- A .csv which holds information like dimensions of every detected object
- The id's: a grayscale png image that show the masks in grayscale
- The masks png that show the masks themselves, in colour
- An overlay that show the masks in colour, the id's and confidence per detected object
- An svg that puts points at the locations of the detected objects

You can also use a folder as input.

```powershell
Windows + Mac: python run.py detect folder ./trainingSet/clear/raw
Linux : sudo venv/bin/python run.py detect folder ./trainingSet/clear/raw
```

Or a folder containing subfolders:

```powershell
Windows + Mac : python run.py detect folders ./trainingSet/folders
Linux : sudo venv/bin/python run.py detect folders ./trainingSet/folders
```

##

## Run by following these detailed steps.

When you want to run every step independently.

The 5 steps are:

- Optionally: create a new model
- Preprocess
- Prepare the dataset
- Train a model with the dataset
- Detect images with a model

##

### Creating a new Model

We'll start by creating a new model that can be used for training.

```powershell
Windows, Mac + Linux : python ./createModel/create_model.py {./trainingModels/new_model.h5}
```

- The first argument is the path to the new model that will be created.

##

### Preprocessing

#### From segments.ai to coco and maskrcnn

In project Preprocessor we can convert multiple segments.ai jsons to a maskrcnn accepted folder structure.

```powershell
Windows, Mac + Linux : python ./preprocessor/preprocess.py ./json/5procent.json ./json/10procent.json crystaltype
```

- The arguments are the jsons you wish to convert
- As last argument you can pass is the name of the folder where the dataset will be generated. This will be a
  ./trainingSet/subfolder.

This will convert and combine the jsons and download every image linked in the jsons into a
./trainingSet/crystaltype/raw folder. The preparation of the dataset will also be arranged. More info about this can be
found at [preparedataset](#preparedatasetpy)

In the ./trainingSet/crystaltype submap you can find the mask coordinates in the masks.json file and the images in the
/raw folder.

##

### Preparation

### prepareDataSet.py

If you already created a set using the preprocessor you can skip this step.

This script creates a train and validation folder for which contains an instances_ids.png and raw.tif file for each
picture. These will be used at the training stage.

You can't run this script if you have not made the aforementioned [filestructure](#dataset-filestructure)

```powershell
Windows, Mac + Linux : python ./maskrcnn/prepareDataSet.py ./trainingSet/clear
```

- The first and only argument is the folder of the crystaltype e.g.: trainingSet/clear.

You can find the output of this script in trainingSet/clear/(train and val).

##

### Model training
If you adhered to the suggested [filestructure](#dataset-filestructure) then use the following command to start
training.

```powershell
Windows + Mac: python ./maskrcnn/train.py ./trainingSet/clear
Linux : sudo venv/bin/python ./maskrcnn/train.py ./trainingSet/clear
```

- The first argument is the main folder of the crystal type.
- The second argument is the weights model you wish to use (default is ./trainingModels/model.h5)

Every Iteration creates a weights model. Before training, please edit the GPU_COUNT according to your machine specs in
maskrcnn/train.py::CrystalConfig class .

A GPU_COUNT of 1 indicates that only the CPU will be used. To also use a GPU set this to > 1.

#### Current best trained model
The current best trained model can be found at ./trainingModels/best_model.h5.
##

### Crystal detection

You can find the output of the detection in the 'out' folder that will be created at the location you provided as the
source. For example: if you provide ./trainingSet/clear, the out folder can be found in: ./trainingSet/clear/raw/out .

A detection will create a few files:

- A .csv which holds information like dimensions of every detected object
- The id's: a grayscale png image that show the masks in grayscale
- The masks png that show the masks themselves, in colour
- An overlay that show the masks in colour, the id's and confidence per detected object
- An svg that puts points at the locations of the detected objects

#### Single image

Only works with .png and .bmp files!

```powershell
Windows + Mac : python ./maskrcnn/detect_tester.py  image  ./trainingSet/clear/raw/IMG_aceton_ (1).png
Linux : sudo venv/bin/python  ./maskrcnn/detect_tester.py  image  ./trainingSet/clear/raw/IMG_aceton_ (1).png
```

- The first argument is the type of detection: image.
- The second argument is the path to the image.
- The third argument the path to the weights model (default is trainingModels/model.h5).

#### One folder with images

```powershell
Windows + Mac : python ./maskrcnn/detect_tester.py  folder  ./trainingSet/clear/raw
Linux : sudo venv/bin/python  ./maskrcnn/detect_tester.py  folder  ./trainingSet/clear/raw
```

- The first argument is the type of detection: folder.
- The second argument is the path to the folder containing the images.
- The third argument the path to the weights model (default is trainingModels/model.h5).

#### Multiple folders with images

Detects all the images in your filestructure. Assuming you adhered to the
suggested [filestructure](#dataset-filestructure).

```powershell
Windows + Mac : python ./maskrcn/detect_tester.py  folders  paths/to/folders
Linux : sudo venv/bin/python  ./maskrcn/detect_tester.py  folders  paths/to/folders
```

- The first argument is the type of detection: folders.
- The second argument is the path to the folders.
- The third argument the path to the weights model (default is trainingModels/model.h5).

### Evaluation
To evaluate predictions in the out/ directory after *detecting*, run evaluate_predictions.py with path to the crystal dir as sys arg (e.g.: ./trainingSet/clear).
This will internally return a list of dictionaries, with each dictionary containing a list of evaluations, this is saved as a massively long json (see evaluations.json inside crystal directory). A single evaluation is either: a prediction with no overlapping mask (false positive), an actual mask with no overlapping prediction (false negative), or a match (overlap of both, true positive, in this case the intersection over union area is included).
Besides this array, each dictionary contains another dictionary with a conf matrix, this is the pixel-level evaluation of the area union of the prediction and the actual mask.
example:
```json
{
    "clear (5)": {
        "crystals": [{
            "id": 5,
            "IOU": 0.75,
            "prediction_polygon": [[25, 36], [26, 37], ...], // coordinates of the predicted crystal polygon, if null -> error has exception
            "actual_polygon": [[22, 39], [21, 36], ...], // coordinates of actual polygon, if null -> error has exception
            "error": null // contains exception message
            }, ...
        ],
        "confusion matrix": {
            "True positive": 0.25, // 25% of the pixels contained by the union of the overlapping predicted and actual mask, is true positive
            "True negative": 0.25,
            "False positive": 0.25,
            "False negative": 0.25
        }
    },
    "clear (6)": ...
}
