import random
import csv
import glob
from pathlib import Path
from slide.slide_container import SlideContainer
from tqdm import tqdm


def create_patches(files, patches_per_slide):
    patches = []
    for i in files:
        patches += patches_per_slide * [i]
    random.shuffle(patches)
    return patches

def load_slides(set, patch_size=256, label_dict=None, level = None, target_folder=None,annotation_file=None, dataset_type=None):
    train_files = []
    valid_files = []
    test_files = []

    with open('/home/klose/Data/datasets.csv', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in tqdm(list(reader)):
            file = glob.glob("{}/**/{}".format(str(target_folder),row["Slide"]), recursive = True)[0]
            if row["Dataset"] == "train" and set.__contains__("train"):
                train_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            elif row["Dataset"] == "val" and set.__contains__("valid"):
                valid_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            elif row["Dataset"] == "test" and set.__contains__("test"):
                test_files.append(SlideContainer(Path(file),annotation_file,level, patch_size, patch_size, dataset_type=dataset_type, label_dict = label_dict))
            else:
                pass

    return train_files, valid_files, test_files

#-----------Debug---------------
#Variablen 
# label_dict = {'Bg':0 , 'Tumor': 1, 'Normal': 2, 'Lumen': -1,
#               'Tub. 1S - Pleo 1': -1, 'Tub. 1S - Pleo 2': -1, 'Tub. 1S - Pleo 3': -1, 
#               'Tub. PluriS - Pleo 1': -1, 'Tub. PluriS - Pleo 2': -1, 'Tub. PluriS - Pleo 3':-1,
#              'Tubulopap. - Pleo 1': -1, 'Tubulopap. - Pleo 2': -1, 'Tubulopap. - Pleo 3': -1, 
#             'Solid - Pleo 1': -1, 'Solid - Pleo 2': -1, 'Solid - Pleo 3': -1,
#              'Nekrose / Entzündung': -1,  'Gefäße - Frei': -1, 'Gefäße - Metastase': -1,
#              'Tumorstroma': -1, 'Ignore': -1, 'Annotationskasten': -1,
#              'Micropap. P1': -1, 'Micropap. P2': -1, 'Micropap. P3': -1,
#              'Kasten - vereinfacht': -1, 'Kasten - Test': -1, 
#               'A1 - B1': -1,'A1 - B2': -1, 'A1 - B3': -1, 'A2 - B1': -1,'A2 - B2': -1,'A2 - B3':-1, 'A3 - B1':-1, 'A3 - B2':-1,'A3 - B3':-1}
# target_folder = "/home/klose/Data/crops"
# annotation_file = "/home/klose/Data/CMC.json"

# train, valid, test = load_slides(["train","valid"], patch_size=2000, label_dict=label_dict, annotation_file=annotation_file, target_folder=target_folder, 
# level = 0, dataset_type='segmentation')

# #save train image 
# train_data = create_patches(train, 1)
# import imageio
# imageio.imwrite("Data/segemted_neu.jpg", train_data[0])