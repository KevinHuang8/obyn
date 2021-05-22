import os
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_xml(filepath):
    '''
    Takes an .xml annotation file with path 'filepath', and returns 
    a tuple (image_name, bboxes), where image_name is the filename
    of the corresponding .tif RGB file, and bboxes are a list of
    [xmin, ymin, xmax, ymax] in pixel values of the ground truth
    bounding boxes for that file.
    '''
    tree = ET.parse(filepath)
    image_name = tree.find('./filename').text
    image_name = Path(image_name).stem
    objects = tree.getroot().findall('.//object')

    bboxes = []
    for tree_obj in objects:
        bbox = tree_obj.find('./bndbox')
        xmin = bbox.find('./xmin').text
        ymin = bbox.find('./ymin').text
        xmax = bbox.find('./xmax').text
        ymax = bbox.find('./ymax').text
        bboxes.append([int(i) for i in [xmin, ymin, xmax, ymax]])

    return image_name, bboxes

def parse_xml_dir(directory):
    data_dict = {}
    for filename in os.listdir(directory):
        image_file, bboxes = parse_xml(directory / filename)
        data_dict[image_file] = bboxes
    return data_dict