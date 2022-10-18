import sys
sys.path.append("../../SlideRunner/")
import numpy as np
import json
from SlideRunner.dataAccess.database import Database
from shapely.geometry import Polygon
import datetime
from tqdm import tqdm

def create_header():
    json_dict = {}
    json_dict["info"] = {
        "description": 'Canine Mammary Tumor',
        "url": "",
        "version": '1.0',
        "year": 2022,
        "contributor": 'Laura Klose, Marc Aubreville, Chloé Puget',
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    json_dict["licenses"] = [{
        "id": 1,
        "name": 'Attribution-NonCommercial-NoDerivs License',
        "url": ''
    }]
    json_dict["categories"] = [
    {
        'id': 1,
        'name': 'Lumen',
        "supercategory": 'Tumor',
    },
    {
        'id': 2,
        'name': 'Tumor',
        'supercategory': 'Tumor',
    },
    {
        'id': 3,
        'name': 'Tub. 1S - Pleo 1',
        'supercategory': 'Tumor',
    },
    {
        'id': 4,
        'name': 'Tub. 1S - Pleo 2',
        'supercategory': 'Tumor',
    },
    {
        'id': 5,
        'name': 'Tub. 1S - Pleo 3',
        'supercategory': 'Tumor',
    },
    {
        'id': 6,
        'name': 'Tub. PluriS - Pleo 1',
        'supercategory': 'Tumor',
    },
    {
        'id': 7,
        'name': 'Tub. PluriS - Pleo 2',
        'supercategory': 'Tumor',
    },
    {
        'id': 8,
        'name': 'Tub. PluriS - Pleo 3',
        'supercategory': 'Tumor',
    },
    {
        'id': 9,
        'name': 'Tubulopap. - Pleo 1',
        'supercategory': 'Tumor',
    },
    {
        'id': 10,
        'name': 'Tubulopap. - Pleo 2',
        'supercategory': 'Tumor',
    },
    {
        'id': 11,
        'name': 'Tubulopap. - Pleo 3',
        'supercategory': 'Tumor',
    },
    {
        'id': 12,
        'name': 'Solid - Pleo 1',
        'supercategory': 'Tumor',
    },
    {
        'id': 13,
        'name': 'Solid - Pleo 2',
        'supercategory': 'Tumor',
    },
    {
        'id': 14,
        'name': 'Solid - Pleo 3',
        'supercategory': 'Tumor',
    },
    {
        'id': 15,
        'name': 'Nekrose / Entzündung',
        'supercategory': 'Tumor',
    },
    {
        'id': 16,
        'name': 'Normal',
        'supercategory': 'Tumor',
    },
    {
        'id': 17,
        'name': 'Gefäße - Frei',
        'supercategory': 'Tumor',
    },
    {
        'id': 18,
        'name': 'Gefäße - Metastase',
        'supercategory': 'Tumor',
    },
    {
        'id': 19,
        'name': 'Tumorstroma',
        'supercategory': 'Tumor',
    },
    {
        'id': 20,
        'name': 'Ignore',
        'supercategory': 'Tumor',
    },
    {
        'id': 21,
        'name': 'Annotationskasten',
        'supercategory': 'Tumor',
    },
    {
        'id': 22,
        'name': 'Micropap. P1',
        'supercategory': 'Tumor',
    },
    {
        'id': 23,
        'name': 'Micropap. P2',
        'supercategory': 'Tumor',
    },
    {
        'id': 24,
        'name': 'Micropap. P3',
        'supercategory': 'Tumor',
    },
    {
        'id': 25,
        'name': 'Kasten - vereinfacht',
        'supercategory': 'Tumor',
    },
    {
        'id': 26,
        'name': 'Kasten - Test',
        'supercategory': 'Tumor',
    },
    {
        'id': 27,
        'name': 'A1 - B1',
        'supercategory': 'Tumor',
    },
    {
        'id': 28,
        'name': 'A1 - B2',
        'supercategory': 'Tumor',
    },
    {
        'id': 29,
        'name': 'A1 - B3',
        'supercategory': 'Tumor',
    },
    {
        'id': 30,
        'name': 'A2 - B1',
        'supercategory': 'Tumor',
    },
    {
        'id': 31,
        'name': 'A2 - B2',
        'supercategory': 'Tumor',
    },
    {
        'id': 32,
        'name': 'A2 - B3',
        'supercategory': 'Tumor',
    },
    {
        'id': 33,
        'name': 'A3 - B1',
        'supercategory': 'Tumor',
    },
    {
        'id': 34,
        'name': 'A3 - B2',
        'supercategory': 'Tumor',
    },
    {
        'id': 35,
        'name': 'A3 - B3',
        'supercategory': 'Tumor',
    }
    ]
    return json_dict



def create_annotation(polygon, image_id, annotation_id, is_crowd):

    segmentation = np.array(polygon["Coords"].exterior.coords).ravel().tolist()

    x, y, max_x, max_y = polygon["Coords"].bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = polygon["Area"]

    annotation = {
        'segmentation': segmentation,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': polygon["Label"],
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_polygon_hierarchy(poly_list):
    for id_outer, outer_poly in poly_list.items():
        for id_inner, inner_poly in poly_list.items():
            if id_outer != id_inner and outer_poly["Coords"].contains(inner_poly["Coords"]):
                outer_poly["Enclosed"].append(id_inner)
                inner_poly["Hierarchy"] += 1



def get_polygon_area(polygon_list, polygon):
    enclosed = 0
#     for poly_within in polygon["Enclosed"]:
#         enclosed += get_polygon_area(polygon_list, polygon_list[poly_within])
    return polygon["Coords"].area - enclosed



def polys_from_sql(database):
    anno_list = []
    image_list = []

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    getslides = """SELECT uid, filename, width, height FROM Slides"""
    for currslide, filename, width, height in tqdm(database.execute(getslides).fetchall()):
        database.loadIntoMemory(currslide)
        image_list.append({'license': 1, 'file_name': filename, 'id': image_id, 'width': width, 'height': height})
        poly_list = {}
        for id, annotation in database.annotations.items():
            if len(annotation.labels) != 0 and annotation.deleted != 1:
                if annotation.annotationType == 3:
                    vector = []
                    poly = {}
                    for x, y in annotation.coordinates:
                        vector.append((int(x), int(y)))

                    poly["Coords"] = Polygon(vector)
                    poly["Label"] = annotation.labels[0].classId
                    poly["Hierarchy"] = 0
                    poly["Enclosed"] = []
                    poly_list[id] = poly

        get_polygon_hierarchy(poly_list)
        poly_list = dict(sorted(poly_list.items(), key=lambda x: x[1]['Hierarchy']))

        for id, poly in poly_list.items():
            area = get_polygon_area(poly_list, poly)
            poly["Area"] = area
            anno_list.append(create_annotation(poly, image_id, annotation_id, is_crowd))
            annotation_id += 1
        image_id += 1

    return image_list, anno_list

def convert(annotation_path):
    database = Database()
    database.open(annotation_path)
    json_dict = create_header()
    image_list, anno_list = polys_from_sql(database)
    json_dict["images"] = image_list
    json_dict["annotations"] = anno_list
    with open('CMC.json', 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    # Define annotation path
    annotation_file = "/home/klose/Data/CMC-21 Klassen-ROI.sqlite"

    # Conversion
    convert(annotation_file)