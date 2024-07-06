import numpy as np
import cv2.cv2 as cv
import json
import os
import sys
import math


def createMasks(data, raw_folder, output_folder, startIndex, applyClahe=False):
    index = startIndex
    for img in data:
        fileName = img["filename"]
        fileFolder = os.path.join(output_folder, "set" + str(index))
        if not os.path.exists(fileFolder):
            os.mkdir(fileFolder)
        index += 1
        fileOutputPath = os.path.join(fileFolder, "instances_ids.png")
        createRegions(img['regions'], fileOutputPath)
        # Copy raw image to set folder
        rawImg = cv.imread(os.path.join(raw_folder, fileName))
        image = cv.cvtColor(rawImg, cv.COLOR_BGR2GRAY)
        if applyClahe:
            print('clahe')
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            print('clahe')
        cv.imwrite(os.path.join(fileFolder, "raw.tif"), image)
    return index


def createRegions(regions, fileOutputPath):
    imgArr = np.zeros((1024, 1500, 1), np.uint8)
    colour = 1
    for region in regions:
        shape_attributes = region["shape_attributes"]
        shapeName = shape_attributes["name"]
        if shapeName == "polygon":
            imgArr = fillPolygon(imgArr, shape_attributes, colour)
        if shapeName == "circle":
            imgArr = fillCircle(imgArr, shape_attributes, colour)
        colour += 1
    cv.imwrite(fileOutputPath, imgArr)


def fillPolygon(imgArr, shape_attributes, colour):
    ptsArr = []
    xPoints = shape_attributes["all_points_x"]
    yPoints = shape_attributes["all_points_y"]
    for i in range(len(xPoints)):
        ptsArr.append([xPoints[i], yPoints[i]])
    pts = np.array(ptsArr, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(imgArr, [pts], colour)
    return imgArr


# TODO LATER
def fillCircle(imgArr, shape_attributes, colour):
    xCoord = shape_attributes['cx']
    yCoord = shape_attributes['cy']
    radius = shape_attributes['r']
    radius = round(radius)
    cv.circle(imgArr, (xCoord, yCoord), radius, colour, -1, 8, 0)
    return imgArr


def prepare_data_set(input_folder):
    raw_folder = input_folder + '/raw'
    train_folder = os.path.join(input_folder, 'train')

    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    val_folder = os.path.join(input_folder, 'val')
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    with open(os.path.join(input_folder, "masks.json"), "r") as read_file:
        dataJson = json.load(read_file)

    # split up in a training and test set but first shuffle the items in the json array because it's too sorted to be
    # effective for training. The val set is saturated by toluene
    np.random.shuffle(dataJson)

    train_len = math.floor(len(dataJson) * 0.8)
    currentIndex = createMasks(dataJson[:train_len], raw_folder, train_folder, 1, False)
    createMasks(dataJson[train_len:], raw_folder, val_folder, currentIndex, False)


if __name__ == '__main__':
    prepare_data_set(sys.argv[1])
