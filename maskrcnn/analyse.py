import os
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import cv2 as cv


def calculate_cell_info(raw_mask, scale):
    features = pd.DataFrame()
    id = 0
    for region in regionprops(raw_mask, intensity_image=raw_mask):
        if region.mean_intensity >= 1:
            # Skip background (intensity 0)
            # Append all features
            features = features.append([get_region_info(region, id, scale)])
            id += 1
    return features


def pixel_to_micrometer(value, scale):
    return value * scale


def get_region_info(region, id, scale):
    # Compute features
    return {'id': id,
            'y': region.centroid[0],
            'x': region.centroid[1],
            'equivalent_diameter': pixel_to_micrometer(region.equivalent_diameter, scale),
            'perimeter': pixel_to_micrometer(max(1, region.perimeter), scale),
            'eccentricity': region.eccentricity * 100,
            'orientation_x_2_sin': np.sin(2 * region.orientation) * 100,
            'orientation_x_2_cos': np.cos(2 * region.orientation) * 100,
            'true_solidity': region.equivalent_diameter / max(1, region.perimeter) * 100,
            'solidity': region.solidity * 100,
            'area': pixel_to_micrometer(region.area, scale),
            'mean_intensity': region.mean_intensity,
            'angle': region.orientation,
            'bbox_top': pixel_to_micrometer(region.bbox[0], scale),
            'bbox_left': pixel_to_micrometer(region.bbox[1], scale),
            'bbox_bottom': pixel_to_micrometer(region.bbox[2], scale),
            'bbox_right': pixel_to_micrometer(region.bbox[3], scale)
            }


def save_to_csv(features, out_path):
    features.to_csv(path_or_buf=out_path, index=False, line_terminator='\n')


def analyseMask(mask, scale, out_path=None):
    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
    features = calculate_cell_info(mask, scale)
    if out_path is not None:
        save_to_csv(features, out_path)
    return features
