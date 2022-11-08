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
        "description": 'Canine Mammary Tumor with new supercategories',
        "url": "",
        "version": '2.0',
        "year": 2022,
        "contributor": 'Laura Klose, Marc Aubreville, Chloé Puget',
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    json_dict["licenses"] = [{
        "id": 1,
        'id_super': 1,
        "name": 'Attribution-NonCommercial-NoDerivs License',
        "url": ''
    }]

    #create categories table 
    json_dict["categories"] = [
    {
        'id': 1,
        'name': 'Lumen',
        "supercategory": 'Tumor',
        'id_super': 1,
    },
    {
        'id': 2,
        'name': 'Tumor',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 3,
        'name': 'Tub. 1S - Pleo 1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 4,
        'name': 'Tub. 1S - Pleo 2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 5,
        'name': 'Tub. 1S - Pleo 3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 6,
        'name': 'Tub. PluriS - Pleo 1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 7,
        'name': 'Tub. PluriS - Pleo 2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 8,
        'name': 'Tub. PluriS - Pleo 3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 9,
        'name': 'Tubulopap. - Pleo 1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 10,
        'name': 'Tubulopap. - Pleo 2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 11,
        'name': 'Tubulopap. - Pleo 3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 12,
        'name': 'Solid - Pleo 1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 13,
        'name': 'Solid - Pleo 2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 14,
        'name': 'Solid - Pleo 3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 15,
        'name': 'Nekrose / Entzündung',
        'supercategory': 'KeinTumor',
        'id_super': 2,
    },
    {
        'id': 16,
        'name': 'Normal',
        'supercategory': 'KeinTumor',
        'id_super': 2,
    },
    {
        'id': 17,
        'name': 'Gefäße - Frei',
        'supercategory': 'KeinTumor',
        'id_super': 2,
    },
    {
        'id': 18,
        'name': 'Gefäße - Metastase',
        'supercategory': 'KeinTumor',
        'id_super': 2,
    },
    {
        'id': 19,
        'name': 'Tumorstroma',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 20,
        'name': 'Ignore',
        'supercategory': 'Other',
        'id_super': -1,
    },
    {
        'id': 21,
        'name': 'Annotationskasten',
        'supercategory': 'Other',
        'id_super': -1,
    },
    {
        'id': 22,
        'name': 'Micropap. P1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 23,
        'name': 'Micropap. P2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 24,
        'name': 'Micropap. P3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 25,
        'name': 'Kasten - vereinfacht',
        'supercategory': 'Other',
        'id_super': -1,
    },
    {
        'id': 26,
        'name': 'Kasten - Test',
        'supercategory': 'Other',
        'id_super': -1,
    },
    {
        'id': 27,
        'name': 'A1 - B1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 28,
        'name': 'A1 - B2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 29,
        'name': 'A1 - B3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 30,
        'name': 'A2 - B1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 31,
        'name': 'A2 - B2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 32,
        'name': 'A2 - B3',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 33,
        'name': 'A3 - B1',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 34,
        'name': 'A3 - B2',
        'supercategory': 'Tumor',
        'id_super': 1,
    },
    {
        'id': 35,
        'name': 'A3 - B3',
        'supercategory': 'Tumor',
        'id_super': 1,
    }
    ]
    return json_dict


#create annotations 
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
        #anpasung der category id, damit diese tumor oder nicht tumo ist 
        #neu: tumor_id
        'tumor_id': polygon["superLabel"],
        'category_id': polygon["Label"],
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

#Order Of Annotations ?
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

#Funktion, um Tumor oder Nicht Tumor Spalte zu erstellen 
def tumor_column_id(class_id): 
    #Tumor_Class: 1-14, 19, 22-24, 27-35 -> 1
    tumor_class = list(range(1,15))
    tumor_class.extend([19, 22, 23, 24])
    tumor_class.extend(list(range(27, 36)))
    #KeinTumor_Class: 15-18 -> 2
    keinTumor_class = list(range(15, 19))
    #Other: 20, 21, 25, 26 -> -1 
    other_class = [20,21,25,26]    
    if class_id in tumor_class: 
        super_id = 1
    if class_id in keinTumor_class: 
        super_id = 2
    if class_id in other_class: 
        super_id = -1
    return super_id



#database = sqlite file
def polys_from_sql(database):
    #make 2 lists for annotations and image paths
    anno_list = []
    image_list = []

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    #get uid, filename, w and h from slides table = getslides df
    getslides = """SELECT uid, filename, width, height FROM Slides"""
    #iterate over getslides df
    for currslide, filename, width, height in tqdm(database.execute(getslides).fetchall()):
        #load image with path 
        database.loadIntoMemory(currslide)
        #append dict with image information to the image_list
        image_list.append({'license': 1, 'file_name': filename, 'id': image_id, 'width': width, 'height': height})

        poly_list = {}
        #iterate over annotations ids of annotations table 
        for id, annotation in database.annotations.items():
            if len(annotation.labels) != 0 and annotation.deleted != 1:
                if annotation.annotationType == 3:
                    vector = []
                    poly = {}
                    #append x,y of the annotation coordinates to vector 
                    for x, y in annotation.coordinates:
                        vector.append((int(x), int(y)))

                    #create poly dict with Coords, LabelClass Id, Hierarchy, Enclosed
                    #here I want to add the superclass Id ??? 
                    poly["Coords"] = Polygon(vector)
                    #neue Kategorie mit SuperID für Tumor Id
                    poly["superLabel"] = tumor_column_id(annotation.labels[0].classId)
                    poly["Label"] = annotation.labels[0].classId
                    poly["Hierarchy"] = 0
                    poly["Enclosed"] = []
                    #add Dict of this annotation to poly_list
                    poly_list[id] = poly

        get_polygon_hierarchy(poly_list)
        poly_list = dict(sorted(poly_list.items(), key=lambda x: x[1]['Hierarchy']))

        #iterate over poly list and get id and poly dict
        for id, poly in poly_list.items():
            #get the area of the polygons
            area = get_polygon_area(poly_list, poly)
            #set new are key with area item 
            poly["Area"] = area
            #make anno_list with poly dict (Coords, LabelClass Id, neu:superLabel, Hierarchy, Enclosed), image id, annotation id and is_crowd
            anno_list.append(create_annotation(poly, image_id, annotation_id, is_crowd))
            annotation_id += 1
        image_id += 1

    return image_list, anno_list

def convert(annotation_path):
    database = Database()
    #open sqlite Database
    database.open(annotation_path)
    json_dict = create_header()

    image_list, anno_list = polys_from_sql(database)
    #append two lists to the json file: one with image_data and one with annotation data can be joined via image id 
    json_dict["images"] = image_list
    json_dict["annotations"] = anno_list


    with open('/home/klose/Data/CMC_Tumor_NonTumor.json', 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    # Define annotation path
    annotation_file = "/home/klose/Data/CMC-21 Klassen-ROI.sqlite"

    # Conversion
    convert(annotation_file)


#mal schauen, hatte da heute ein paar Änderungen gemacht 