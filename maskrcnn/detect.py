import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import cv2.cv2 as cv2
import fnmatch
import math
import maskrcnn.analyse as analyse
import maskrcnn.overlay as overlay
from maskrcnn.modelClasses.crystal_inference_config import CrystalInferenceConfig
import drawSvg as draw
import colorsys
from tqdm import tqdm
from maskrcnn.modelClasses.dot import Dot
from maskrcnn.train import CrystalConfig
from maskrcnn.mrcnn import model as modellib
from random import shuffle


def detect(self, img_path, model_name, scale):
    # Get image path without extension
    img_dir = os.path.dirname(img_path)
    img_name_no_extension = os.path.basename(img_path).split('.')[0]
    paths = defineOutputPaths(img_dir, img_name_no_extension, model_name)
    # Load image
    image = cv2.imread(img_path)
    features = startDetection(self, image, paths, scale)
    mask = cv2.imread(paths['mask_ids_out_path'])
    analyse.save_to_csv(features, paths['features_out_path'])
    # Generate array of distinct colour for visualization
    colors = generate_distinct_colors(n_colors=features.shape[0])
    # Get ids of mask to convert to colour later
    mask_ids = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    # Initialize color mask and apply colors
    color_mask = getColorMask(mask, mask_ids, paths, colors)
    # Overlay mask on original image
    image_with_overlay = overlay.blend_transparent(image, color_mask)
    createSVG(image, image_with_overlay, features, paths)


def defineOutputPaths(img_dir, img_name, model_name):
    output_base_path = os.path.join(img_dir, 'out')
    output_image_path = os.path.join(output_base_path, img_name)
    paths = {
        'output_base_path': output_base_path,
        'output_image_path': output_image_path,
        'features_out_path': os.path.join(output_image_path, img_name + '.csv'),
        'mask_ids_out_path': os.path.join(output_image_path, img_name + '_ids.png'),
        'mask_out_path': os.path.join(output_image_path, img_name + '_mask.png'),
        'overlay_out_path': os.path.join(output_image_path, img_name + '_overlay.png'),
        'svg_points_path': os.path.join(output_image_path, img_name + '_points.svg'),
        'stats_path': os.path.join(output_base_path, 'stats.csv'),
        'used_model_path': os.path.join(output_image_path, 'model.txt')
    }
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)

    modelFile = open(paths['used_model_path'], 'w')
    modelFile.write(model_name)
    modelFile.close()

    return paths


def startDetection(self, image, paths, scale):
    r = self.model.detect([image], verbose=0)[0]
    scores = r['scores']
    avg_score = np.mean(scores)
    if self.accuracy == -1:
        self.accuracy = avg_score
    else:
        self.accuracy = (avg_score + self.accuracy) / 2
    saveMask(r, paths)
    features = extractMaskFeatures(cv2.imread(
        paths['mask_ids_out_path']), scores, scale)
    return features


def saveMask(r, paths):
    if r['masks'].shape[2] == 0:
        mask = np.zeros(
            (r['masks'].shape[0], r['masks'].shape[1], 1), np.float64)
    else:
        mask = np.argmax(r['masks'], 2)
    cv2.imwrite(paths['mask_ids_out_path'], mask)


def extractMaskFeatures(mask, scores, scale):
    features = analyse.analyseMask(mask, scale)
    features['score'] = scores[:features.shape[0]] * 100
    return features


def getColorMask(mask, mask_ids, paths, colors):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2] + 1))
    colour_index = 0
    for color in mask_ids:
        if not (color == [0, 0, 0]).all():
            instance = cv2.inRange(mask, color, color)
            color_mask[np.where(instance > 0)] = colors[colour_index]
            colour_index += 1
    cv2.imwrite(paths['mask_out_path'], color_mask)
    return cv2.imread(paths['mask_out_path'], -1)


def createSVG(image, image_with_overlay, features, paths):
    font = cv2.FONT_HERSHEY_DUPLEX
    d = draw.Drawing(image.shape[1], image.shape[0],
                     origin=(0, -image.shape[0]))
    for index, row in features.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        height = image_with_overlay.shape[0]
        width = image_with_overlay.shape[1]
        dot = Dot(row['id'])
        dot.append(draw.Circle(x, -y, 10, fill='red',
                               stroke_width=2, stroke='black'))
        d.append(dot)
        if y > height - 20:
            y = height - 30
        if x < 20:
            x = 30
        if x > width - 100:
            x = width - 150
        if y < 20:
            y = 30
        cv2.putText(image_with_overlay, str(int(row['id'])) + " ("
                    + str(truncate(row['score'], 2)) + ")", (x, y), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    d.saveSvg(paths['svg_points_path'])
    cv2.imwrite(paths['overlay_out_path'], image_with_overlay)


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def generate_distinct_colors(n_colors, n_intensity_levels=2, max_channel_val=255, background_color=None):
    n_colors_per_intensity = int(np.ceil(n_colors / n_intensity_levels))
    RGB_tuples = []
    for intensity in np.linspace(1.0, 0.0, n_intensity_levels):
        HSV_tuples = [(x / n_colors_per_intensity,
                       1,
                       1 - intensity / n_intensity_levels)
                      for x in range(n_colors_per_intensity)]
        RGB_tuples.extend([(int(np.floor(0.5 + x[0] * max_channel_val)),
                            int(np.floor(0.5 + x[1] * max_channel_val)),
                            int(np.floor(0.5 + x[2] * max_channel_val)),
                            100)
                           for x in list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))])

    shuffle(RGB_tuples)
    if background_color is not None:
        # Make sure the desired background color is the first one in the list
        if background_color in RGB_tuples:
            RGB_tuples.remove(background_color)
        RGB_tuples.insert(0, background_color)
    return RGB_tuples


class Detect:
    def __init__(self, model_path):
        config = CrystalInferenceConfig()  # CellInferenceConfig()

        model_name = os.path.basename(model_path)
        model_name = os.path.splitext(model_name)[0]

        self.model = modellib.MaskRCNN(
            mode="inference", config=config, model_dir='./output', model_name=model_name)
        self.model.load_weights(model_path, by_name=True)
        self.accuracy = -1

    def detectSingle(self, img_path, model_name, scale):
        detect(self, img_path, model_name, scale)
        print('Average confidence: ', self.accuracy)

    def detectMultiple(self, img_dir, model_name, scale):
        count = len(fnmatch.filter(os.listdir(img_dir), '*.bmp')) + \
                len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        pbar = tqdm(total=count)  # progressbar in console
        print('\nAnalysing ' + str(count) + ' images...')
        for filename in os.listdir(img_dir):
            if filename.lower().endswith((".bmp", '.png')):
                pbar.set_description(filename)
                img_path = os.path.join(img_dir, filename)
                self.detectSingle(img_path, model_name, scale)
                pbar.update(1)
        print('Average confidence: ', self.accuracy)
