import sys
import warnings
import cv2.cv2 as cv2
from shapely.errors import ShapelyError
from shapely.geometry import Polygon, GeometryCollection, Point, MultiPolygon
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from project_utils import ROOT_DIR
import pandas as pd


def format_json_mask(mask):
    mask_polygons = []
    for region in mask["regions"]:
        coordinates = []
        xcoords = region["shape_attributes"]["all_points_x"]
        ycoords = region["shape_attributes"]["all_points_y"]

        if len(xcoords) <= 2:
            continue

        for i in range(len(xcoords)):
            coordinates.append([xcoords[i], ycoords[i]])

        mask_polygons.append(coordinates)
    return mask_polygons


def construct_evaluation(id, overlap, p, a, error=None):
    return {"prediction_id": id, "IOU": overlap, "prediction_polygon": p, "actual_polygon": a, "error": error}


def pre_process_with_has_matched_boolean(predictions, actuals, coordinates):
    predicted_polygons = []
    actual_polygons = []

    for prediction in predictions:
        center = []
        for coord in coordinates:
            # a polygon can only be drawn from minimum 3 points!
            if len(prediction) > 2 and Polygon(prediction).contains(Point([coord[1], coord[2]])):
                center = coord
                break

        # this is een array of:
        # boolean: shows if this prediction polygon overlaps with an actual mask
        # array of points: coordinates that form the polygon
        # center: [id, x, y]
        if len(prediction) > 2:
            predicted_polygons.append([False, prediction, center])

    for a in actuals:
        actual_polygons.append([False, a])

    return predicted_polygons, actual_polygons


def evaluate_mask(predictions, actuals, coordinates):
    """
    1: preprocess the actual masks and predicted masks
    2: for every actual mask, look for an overlapping predicted mask
        -> if any of them can't be converted to a polygon (because of self-intersection etc.) then make an evaluation and specify the exception message.
    3: if we reach an actual mask and a predicted mask that overlap each other to some extent, calculate that overlapping area and make make an evaluation with metric data.
    4: if there are actual masks or predicted masks that aren't tagged to have been matched with anyone,
        then evaluate them as 'lonely' (make an evaluation with only 1 polygon and no metric data)
    """
    evaluations = []

    # this function does 2 things
    # 1: link coordinates and id from the metrics csv, to the prediction polygons
    # 2: place all the actual and predicted polygons in an array with a boolean that tells if that polygon already is matched with another (that overlaps)
    predicted_polygons, actual_polygons = pre_process_with_has_matched_boolean(predictions, actuals, coordinates)

    # we start by iterating the actual masks, and see if they overlap with any of the predictions
    # actual[1] contains the array of coordinates that form the actual mask polygon
    for actual in actual_polygons:
        if actual[0] is True:
            continue
        try:
            # test if actual[1] forms a valid polygon
            # this should never fail if the actual masks are well annotated
            intersection_over_union(Polygon(actual[1]), Polygon(actual[1]))
        except ShapelyError as e:
            # if we catch a shapely error, create an evaluation but specify the error message (these are the ones with 'an error')
            evaluations.append(construct_evaluation(None, None, None, actual[1], str(e)))
            actual[0] = True  # avoid revisiting this malformed polygon later on
            continue

        # prediction[1] contains the array of coordinates that form the predicted mask polygon
        for prediction in predicted_polygons:
            # We don't want to revisit polygons that already are matched with an actual mask
            if prediction[0] is True:
                continue
            try:
                overlap_percentage = intersection_over_union(Polygon(prediction[1]), Polygon(actual[1]))
            except ShapelyError as e:
                # if we catch a shapely error, create an evaluation but specify the error message (these are the ones with 'an error')
                _id = None
                if len(prediction[2]) != 0:
                    _id = prediction[2][0]
                evaluations.append(construct_evaluation(_id, None, prediction[1].tolist(), None, str(e)))
                prediction[0] = True  # avoid revisiting this polygon later on
                continue

            if overlap_percentage > 0:
                # We get to this point if we reached an actual mask and a predicted mask that overlap to some extent
                actual[0] = prediction[0] = True
                _id = None
                if len(prediction[2]) != 0:
                    _id = prediction[2][0]
                evaluations.append(construct_evaluation(_id, overlap_percentage, prediction[1].tolist(), actual[1]))

        # for actual masks that don't overlap with any prediction:
        if actual[0] is False:
            evaluations.append(construct_evaluation(None, 0, None, actual[1]))

    # for predictions that don't overlap with any actual mask
    for prediction in predicted_polygons:
        if prediction[0] is False:
            evaluations.append(construct_evaluation(None, 0, prediction[1].tolist(), None))

    return evaluations


def intersection_over_union(polygon1, polygon2):
    """
    this returns the overlap percentage, 2 the same polygons return 1.0.
    Takes in 2 (x, 2)-dimensional arrays
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1.intersection(polygon2).area
    polygon_union = polygon1.area + polygon2.area - polygon_intersection

    return polygon_intersection / polygon_union


def draw_evaluations(all_evaluations, crystal_dir):
    multipoly_counter = 0
    pbar = tqdm(total=len(all_evaluations))
    for crystal_key in all_evaluations:
        pbar.set_description(crystal_key)
        clean_png = cv2.cvtColor(cv2.imread("{}/raw/{}.png".format(crystal_dir, crystal_key), 0), cv2.COLOR_GRAY2BGR)

        actuals = []
        predictions = []
        overlappings = []
        for crystal in all_evaluations[crystal_key]["crystals"]:
            actual_exists = crystal["actual_polygon"] is not None
            prediction_exists = crystal["prediction_polygon"] is not None
            overlapping_areas = 0
            overlapping_coordinates = []

            if actual_exists and prediction_exists:
                overlapping_areas = Polygon(crystal["prediction_polygon"]).intersection(
                    Polygon(crystal["actual_polygon"]))

            if actual_exists:
                actuals.append(np.expand_dims(crystal["actual_polygon"], 1))

                if prediction_exists:
                    if isinstance(overlapping_areas, GeometryCollection):
                        for area in overlapping_areas:
                            if not isinstance(area, Polygon):
                                continue
                            x, y = np.array(Polygon(area).exterior.coords.xy)

                            for i in range(len(x)):
                                overlapping_coordinates.append([x[i], y[i]])
                            overlappings.append(np.expand_dims(overlapping_coordinates, 1).astype(int))
                    elif overlapping_areas.geom_type == 'Polygon':
                        predictions.append(np.expand_dims(crystal["prediction_polygon"], 1))
                        x, y = np.array(overlapping_areas.exterior.coords.xy)
                        for i in range(len(x)):
                            overlapping_coordinates.append([x[i], y[i]])
                        overlappings.append(np.expand_dims(overlapping_coordinates, 1).astype(int))
                    elif overlapping_areas.geom_type == 'MultiPolygon':
                        # TODO: what do we do with nested polygons, stacked upon each other?
                        # print("Encountered a nested MultiPolygon, not drawing it, TODO")
                        multipoly_counter += 1

            elif prediction_exists:
                predictions.append(np.expand_dims(crystal["prediction_polygon"], 1))

        # draw the actual annotated masks [BLUE]
        cv2.drawContours(clean_png, actuals, -1, (224, 79, 72), 1)  # leave at -1! This is an index (draw all polygons)
        # draw the predicted masks [GREEN]
        cv2.drawContours(clean_png, predictions, -1, (50, 168, 82), 1)
        # draw the overlapping areas [RED]
        cv2.drawContours(clean_png, np.array(overlappings), -1, (227, 38, 0), 1)

        # print("writing {}/raw/out/{}/{}_intersections.png".format(crystal_dir, crystal_key, crystal_key))
        cv2.imwrite("{}/raw/out/{}/{}_intersections.png".format(crystal_dir, crystal_key, crystal_key), clean_png)
        pbar.update(1)

    print(f'\nEncountered {multipoly_counter} Multipolygons that could not be drawn.')


def conf_matrix(_eval, im_surface):
    tp = 0
    fp = 0
    fn = 0
    for item in _eval["crystals"]:
        if item["prediction_polygon"] is None:
            if item["actual_polygon"] is not None:
                fn += Polygon(item["actual_polygon"]).area
        elif item["actual_polygon"] is None:
            fp += Polygon(item["actual_polygon"]).area
        else:
            tp += Polygon(item["prediction_polygon"]).intersection(Polygon(item["actual_polygon"])).area
            fn += Polygon(item["actual_polygon"]).area - item["IOU"]
            fp += Polygon(item["prediction_polygon"]).area - item["IOU"]
    tn = im_surface - tp - fp - fn

    appel = {"True positive": tp / im_surface, "False positive": fp / im_surface, "True negative": tn / im_surface,
            "False negative": fn / im_surface}

    return appel


def print_evaluation_metrics(all_evaluations):
    total_IOU = 0
    iou_count = 0
    # whether or not to calculate avg IOU only from matches
    exclude_zeros = True

    matches_only = ""
    if exclude_zeros:
        matches_only = "(including only matches)"

    avg_tp = 0
    avg_fp = 0
    avg_tn = 0
    avg_fn = 0

    all_matches = 0
    total_amount_of_evaluations = 0
    image_counter = 0

    all_predictions = 0
    all_predictions_with_error = 0
    all_predictions_with_no_mask = 0

    all_actual_masks = 0
    all_actual_masks_with_error = 0
    all_actual_masks_with_no_prediction = 0

    # all_evaluations contains an array of dictionaries, each dictionary represents 1 crystal PNG.
    # This dictionary contains an array of evaluations of that specific crystal PNG.
    # Every evaluation inside this dictionary is either an actual mask without a matching prediction, a prediction without a matching actual mask,
    # or 1 overlapping of both (a match with an IOU and id etc.)
    for crystal_key in all_evaluations:
        png = all_evaluations[crystal_key]
        avg_tp += png["confusion_matrix"]["True positive"]
        avg_fp += png["confusion_matrix"]["False positive"]
        avg_tn += png["confusion_matrix"]["True negative"]
        avg_fn += png["confusion_matrix"]["False negative"]

        image_counter += 1

        for crystal in png["crystals"]:
            total_amount_of_evaluations += 1
            if crystal["error"] is not None:
                if crystal["actual_polygon"] is not None:
                    all_actual_masks_with_error += 1
                else:
                    all_predictions_with_error += 1
                continue

            if crystal["prediction_polygon"] is not None:
                all_predictions += 1
                if crystal["actual_polygon"] is None:
                    all_predictions_with_no_mask += 1
                else:
                    all_matches += 1

            elif crystal["actual_polygon"] is not None:
                all_actual_masks += 1
                all_actual_masks_with_no_prediction += 1

            iou = crystal["IOU"]  # exclude => !0 (logical implication)
            if iou is not None and (not exclude_zeros or iou != 0):
                total_IOU += iou
                iou_count += 1

    print("The average Intersection over Union is {} percent. {}".format(total_IOU / iou_count, matches_only))
    print("{}/{} evaluations had errors and were not evaluated.".format(
        all_predictions_with_error + all_actual_masks_with_error, total_amount_of_evaluations))
    # print("{}/{} masks with no prediction".format(all_actual_masks_with_no_prediction, total_amount_of_evaluations - (
    #         all_predictions_with_error + all_actual_masks_with_error)))
    # print("{}/{} predictions with no mask".format(all_predictions_with_no_mask, total_amount_of_evaluations - (
    #         all_predictions_with_error + all_actual_masks_with_error)))
    # print("{}/{} matches (overlapping)".format(all_matches, total_amount_of_evaluations - (
    #         all_predictions_with_error + all_actual_masks_with_error)))

    # now print the averaged confusion matrix:
    print(f"Pixel-level confusion matrix, averaged over {len(all_evaluations)} individual pictures:")
    print(" |   T   |   F   |")
    print("T|%6.2f%%|%6.2f%%|" % (avg_tp * 100 / image_counter, avg_fp * 100 / image_counter))
    print("F|%6.2f%%|%6.2f%%|\n" % (avg_fn * 100 / image_counter, avg_tn * 100 / image_counter))


def evaluate(crystal_dir):
    # get all crystal subdirectories
    path, subdirs, files = next(os.walk(crystal_dir + "/raw/out/"))

    # load all the actual masks: masks.json
    file = open(crystal_dir + "/masks.json", encoding='utf-8')
    json_masks = json.load(file)
    file.close()

    all_evaluations = {}
    for subdir in subdirs:
        # read the predicted mask.png masks
        file_path = "{}/raw/out/{}/{}_mask.png".format(crystal_dir, subdir, subdir)
        predicted_masks = cv2.imread(file_path)

        # find the actual crystal masks in the json which corresponds to the current crystal subdir
        # (the crystal masks of the current PNG in this subdir)
        target_mask = {}
        for mask in json_masks:
            # subdir is actually the crystal filename without file extension
            if mask["filename"].split('.')[0] == subdir:
                target_mask = mask

        # A) convert the actual json masks to a 2 dim array of coordinates that each form a polygon
        all_actual_masks = format_json_mask(target_mask)

        # B) convert the predicted masks to a 2 dim array of coordinates that each form a polygon
        grey_scaled = cv2.cvtColor(predicted_masks, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey_scaled, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        predicted_polygons = []
        for c in contours:
            predicted_polygons.append(np.array(np.squeeze(c)))

        # C) get the coordinates of the predicted crystals out of the metrics csv
        try:
            df = pd.read_csv("{}/raw/out/{}/{}.csv".format(crystal_dir, subdir, subdir))
            coordinates = df[['id', 'x', 'y']].to_numpy()
        except KeyError as e:
            warnings.warn(
                f"It looks like {subdir}'s metrics csv is malformed, skipping this crystal directory\nexception: {e}")
            continue

        # Now the predicted and actual masks are in a 2 dim array of coordinates that each form a polygon.
        # This will be used to convert to polygons and calculate overlap percentage
        _eval = {"crystals": evaluate_mask(predicted_polygons, all_actual_masks, coordinates)}

        # Now that we have IOU and data of the crystals,
        # calculate confusion matrix on pixel-level for this crystal PNG
        _eval["confusion_matrix"] = conf_matrix(_eval, predicted_masks.shape[0] * predicted_masks.shape[1])

        # all_evaluations is an array of dictionaries, each dictionary contains a list of evaluations.
        # every such evaluation is another dictionary that contains the overlap percentage, actual mask polygon, predicted mask polygon etc.
        # so: an array for all crystal dirs, every crystal dir has a PNG with multiple predictions and actual mask, every such prediction or actual mask has an evaluation
        all_evaluations[subdir] = _eval

    draw_evaluations(all_evaluations, crystal_dir)

    # get average intersection over union:
    print_evaluation_metrics(all_evaluations)
    with open(f"{crystal_dir}/evaluations.json", 'w') as file:
        file.write(json.dumps(all_evaluations, indent=4))

    return all_evaluations


if __name__ == "__main__":
    # path to the crystal dir
    # e.g.: /home/$USER/Desktop/usiigaci-optimized/postprocessing/data/clear
    if len(sys.argv) >= 2:
        crystal_dir = sys.argv[1]
    else:
        crystal_dir = ROOT_DIR + "/trainingSet/clear/"

    evaluate(crystal_dir)
