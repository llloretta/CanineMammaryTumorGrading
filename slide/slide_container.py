import openslide
import cv2
from fastai.vision import *
from shapely import geometry

#neueste Anpassung, damit ich Tumor und nicht Tumor trainieren kann ist der self.name der davor "name" war 
class SlideContainer:

    def __init__(self, file: Path,
                 annotation_file,
                 level: int = 0,
                 width: int = 256, height: int = 256,
                 sample_func=None, dataset_type=None, label_dict=None):
        #Parameter, um einzustellen aus welcher Spalte die Klassennamen gewählt werden supercategory = Tumor/ NichtTumor + id_super
        #klasse = "supercategory"
        self.poly_klasse = "tumor_id" #old: "category_id"
        #Problem: die Annotationen und Klassen sollen erst gejoint werden, wofür man die Ids braucht, dann soll aber nur noch die super_id verwendet werden 
        self.id = 'id_super' #1 = Tumor, 2= KeinTumor, 3 = Other
        self.name = 'supercategory'

        self.file = file
        with open(annotation_file) as f:
            data = json.load(f)


            # tissue classes ist dictionary mit allen Klassen in den Daten als key und value die id: self tissue classes {'Lumen': 1, 'Tumor': 2, 'Tub. 1S - Pleo 1': 3 ... 
            # self.tissue_classes = dict(zip([cat["name"] for cat in data["categories"]], [
            #                            cat["id"] for cat in data["categories"]]))
            #from json file get table categories, that has 'id', 'name', 'supercategory' and 'id_super'

            #{'Lumen': 1, 'Tumor': 1, 'Tub. 1S - Pleo 1': 1, hier schon die Klasse Tumor zurodnen 
            self.tissue_classes = dict(zip([cat[self.name] for cat in data["categories"]], [
                                       cat[self.id] for cat in data["categories"]]))
       
            
            #get the image_id  
            image_id = [i['id']
                        for i in data["images"] if i["file_name"] == file.name][0]

            #get the polygons of the image with image_id 
            self.polygons = [anno for anno in data['annotations']
                             if anno["image_id"] == image_id]
  
        #labels ist Liste mit den enthaltenen Klassen IDs im jeweiligen Bild (ungefiltert)
        #labels {2, 13, 14, 15, 17, 19, 20} hier sind die originalen Ids, die ich auch noch brauche für die zuordnung der koordinaten ?! echt, eigentlich kann ich die auch rasmachen
        #self.labels = set([poly["category_id"] for poly in self.polygons])
        #tumor labels 
        self.labels = set([poly[self.poly_klasse] for poly in self.polygons])
        # print("labels",self.labels)
        
        
        
        #hier ungenutzte Label entfernen 
        #         self.labels.discard(self.tissue_classes["Bone"])
        #         self.labels.discard(self.tissue_classes["Cartilage"])
        if self.name != "supercategory":
            self.labels.discard(self.tissue_classes["Annotationskasten"])
            self.labels.discard(self.tissue_classes["Ignore"])
            self.labels.discard(self.tissue_classes["Kasten - vereinfacht"])
            self.labels.discard(self.tissue_classes["Kasten - Test"])
        

        #training_dict: {2: None, 13: None, 14: None, 15: None, 17: None, 19: None, 20: None} neu: {1: None, 2: None, -1: None}
        self.training_dict = dict.fromkeys(list(self.labels))
        #print("list labels", self.labels)
        
        
        #initiale Wahrscheinlichkeiten sind alle gleich  0.14285714285714285
        self.probabilities = dict.fromkeys(
            list(self.labels), 1 / len(list(self.labels)))
    
        # Slide einlesen
        self.slide = openslide.open_slide(str(file))
        #print("slide dimensions", self.slide.dimensions)
        # Graubild erstellen
        thumbnail = cv2.cvtColor(
            np.array(self.slide.read_region((0, 0), self.slide.level_count - 1, self.slide.level_dimensions[-1]))[:, :,
                                                                                                                  :3], cv2.COLOR_RGB2GRAY)

        # # Debuggin
        # import imageio
        # thumbnail_2 = cv2.cvtColor(
        #     np.array(self.slide.read_region((0, 0), self.slide.level_count - 1, self.slide.level_dimensions[-1]))[:, :,
        #     :3], cv2.COLOR_RGB2GRAY)
        # print("thumbnail")
        # #Graubild
        # imageio.imwrite("Data/segemted_neu.jpg", thumbnail_2)
        # # plt.imshow(thumbnail)
        # # Ende

        blurred = cv2.GaussianBlur(thumbnail, (5, 5), 0)
        self.white, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]

        if level is None:
            level = self.slide.level_count - 1
        self._level = level
        self.sample_func = sample_func
        self.dataset_type = dataset_type
        # das definierte label dict
        self.label_dict = label_dict

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_new_level(self):
        return self._level

    # FUnktion um Region aus der Slide zu lesen
    def get_patch(self, x: int = 0, y: int = 0):
        rgb = np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                              level=self._level, size=(self.width, self.height)))[:, :, :3]
        return rgb

    # Funktion um die Konturen zu zeichnen
    def get_y_patch(self, x: int = 0, y: int = 0):  #HIER LÄUFT WAHRSCHEINLICH WAS SCHIEF
        # leere Segmentierungsmaske erstellen
        y_patch = -1*np.ones(shape=(self.height, self.width), dtype=np.int8)
        #inv_map ist map mit allen Classen + ids: {1: 'Lumen', 2: 'Tumor', 3: 'Tub. 1S - Pleo 1', 4: 'Tub....
        #print("tissue classes", self.tissue_classes) #PASST
        inv_map = {v: k for k, v in self.tissue_classes.items()}  #hier passt es nicht ... es wird ein Dictionary mit den Klassen erstellt und da die Namen nicht gleich sind, werden nicht all betrachetet
        #print(inv_map)

        # über die Polygone iterieren - eine Liste mit Json Formatierung der Annotationen 
        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape(
                (-1, 2)) / self.down_factor
            # coordinates für patch anpassen
            coordinates = coordinates - (x, y)
            # label ist der zugeordnete Wert im dict ? 
            # aus Poly die Id auslesen, in der inv passenden Klassennamen findne, und im Label dict schauen, welcher Wert gesetzt ist 
            # label = self.label_dict[inv_map[poly["category_id"]]]
            label = self.label_dict[inv_map[poly[self.poly_klasse]]] #HIER 
            # zeichnen der Konturen auf das y_Patch = image (img, contours, -1 = all contours, label, color)
            cv2.drawContours(y_patch, [coordinates.reshape(
                (-1, 1, 2)).astype(int)], -1, label, -1)

        # white mask for parts without annotation
        if self.dataset_type == 'segmentation':
            white_mask = cv2.cvtColor(self.get_patch(
                x, y), cv2.COLOR_RGB2GRAY) > self.white
            excluded = (y_patch == -1)
            # white mask, patch und y_patch zusammen
            y_patch[np.logical_and(white_mask, excluded)] = 0                        
        return y_patch

    def get_new_train_coordinates(self):
        inv_map = {v: k for k, v in self.tissue_classes.items()}
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.polygons, **{"classes": self.labels, "size": self.shape,
                                                      "level_dimensions": self.slide.level_dimensions,
                                                      "level": self.level})
        # default sampling method
        xmin, ymin = 0, 0
        found = False
        while not found:
            iter = 0
            label = random.choices(list(self.probabilities.keys()), list(
                self.probabilities.values()))[0]
            polygons = [
                poly for poly in self.polygons if poly[self.poly_klasse] == label]
                # poly for poly in self.polygons if poly["category_id"] == label]
            polygons_area = [poly["area"] for poly in polygons]
            polygons_area = np.array(polygons_area) / sum(polygons_area)
            polygon = random.choices(polygons, polygons_area)[0]
            coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
            # Box mit dem Bereich der Annotationsklasse
            minx, miny, xrange, yrange = polygon["bbox"]
            while iter < 25 and not found:
                # geht hier rein
                iter += 1
                # aus dem Bereich von min - max sollen für x und y Koordinaten generiert werden
                # Änderung
                # pnt = geometry.Point(random.uniform(
                #     minx + self.width, minx + xrange), random.uniform(miny + self.height, miny + yrange))
                pnt = geometry.Point(random.uniform(minx, minx + xrange), random.uniform(miny, miny + yrange))
                #pnt = geometry.Point(random.uniform(minx, xrange), random.uniform(miny, yrange))
                # Ende
                if geometry.Polygon(coordinates).contains(pnt):
                    # vom random x und y Wert (aus originalem Level (des. noch durch down Faktor teilen ohne Rest)) werden Breite und Höhe abgezogen
                    xmin = pnt.x // self.down_factor - self.width  #/ 2
                    ymin = pnt.y // self.down_factor - self.height  #/ 2

                    #falls die x oder y Werte negativ sind 
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0

                    found = True
                if self.dataset_type == 'classification' and found:
                    if np.unique(self.get_y_patch(xmin, ymin))[
                        np.argmax(np.unique(self.get_y_patch(xmin, ymin), return_counts=True)[1])] != self.label_dict[
                        inv_map[label]] or np.sum(self.get_y_patch(xmin, ymin) == self.label_dict[inv_map[label]]) < (
                            self.width * self.height * 0.9):
                        found = False
        return xmin, ymin

# original von Frauke
    # def get_new_train_coordinates(self):
    #     inv_map = {v: k for k, v in self.tissue_classes.items()}
    #     # use passed sampling method
    #     if callable(self.sample_func):
    #         return self.sample_func(self.polygons, **{"classes":self.labels ,"size": self.shape,
    #                                             "level_dimensions": self.slide.level_dimensions,
    #                                             "level": self.level})
    #     # default sampling method
    #     xmin, ymin = 0,0
    #     found = False
    #     while not found:
    #         iter = 0
    #         label = random.choices(list(self.probabilities.keys()), list(self.probabilities.values()))[0]
    #         polygons = [poly for poly in self.polygons if poly["category_id"] == label]
    #         polygons_area = [poly["area"] for poly in polygons]
    #         polygons_area = np.array(polygons_area) / sum(polygons_area)
    #         polygon = random.choices(polygons, polygons_area)[0]
    #         coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
    #         minx, miny, xrange, yrange = polygon["bbox"]
    #         while iter < 25 and not found:
    #             iter += 1
    #             pnt = geometry.Point(random.uniform(minx, minx + xrange), random.uniform(miny, miny + yrange))
    #             if geometry.Polygon(coordinates).contains(pnt):
    #                 xmin = pnt.x // self.down_factor - self.width / 2
    #                 ymin = pnt.y // self.down_factor - self.height / 2
    #                 found = True
    #             if self.dataset_type == 'classification' and found:
    #                 if np.unique(self.get_y_patch(xmin, ymin))[
    #                     np.argmax(np.unique(self.get_y_patch(xmin, ymin), return_counts=True)[1])] != self.label_dict[
    #                     inv_map[label]] or np.sum(self.get_y_patch(xmin, ymin) == self.label_dict[inv_map[label]]) < (
    #                         self.width * self.height * 0.9):
    #                     found = False
    #     return xmin, ymin

    # def __str__(self):
    #     return str(self.file)


# save copy
    # def get_new_train_coordinates(self):
    #     inv_map = {v: k for k, v in self.tissue_classes.items()}
    #     # use passed sampling method
    #     if callable(self.sample_func):
    #         return self.sample_func(self.polygons, **{"classes":self.labels ,"size": self.shape,
    #                                            "level_dimensions": self.slide.level_dimensions,
    #                                            "level": self.level})
    #     # default sampling method
    #     xmin, ymin = 0,0
    #     found = False
    #     while not found:
    #         iter = 0
    #         label = random.choices(list(self.probabilities.keys()), list(self.probabilities.values()))[0]
    #         polygons = [poly for poly in self.polygons if poly["category_id"] == label]
    #         polygons_area = [poly["area"] for poly in polygons]
    #         polygons_area = np.array(polygons_area) / sum(polygons_area)
    #         polygon = random.choices(polygons, polygons_area)[0]
    #         coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
    #         #Box mit dem Bereich der Annotationsklasse
    #         minx, miny, xrange, yrange = polygon["bbox"]
    #         while iter < 25 and not found:
    #             #geht hier rein
    #             iter += 1
    #             #aus dem Bereich von min - max sollen für x und y Koordinaten generiert werden
    #             # Änderung
    #             pnt = geometry.Point(random.uniform(minx + self.width, minx + xrange), random.uniform(miny+ self.height, miny + yrange))
    #             #pnt = geometry.Point(random.uniform(minx, xrange), random.uniform(miny, yrange))
    #             #Ende
    #             if geometry.Polygon(coordinates).contains(pnt):
    #                 #vom random x und y Wert werden Breite und Höhe abgezogen
    #                 xmin = pnt.x // self.down_factor - self.width  # / 2
    #                 ymin = pnt.y // self.down_factor - self.height # / 2

    #                 found = True
    #             if self.dataset_type == 'classification' and found:
    #                 if np.unique(self.get_y_patch(xmin, ymin))[
    #                     np.argmax(np.unique(self.get_y_patch(xmin, ymin), return_counts=True)[1])] != self.label_dict[
    #                     inv_map[label]] or np.sum(self.get_y_patch(xmin, ymin) == self.label_dict[inv_map[label]]) < (
    #                         self.width * self.height * 0.9):
    #                     found = False
    #     return xmin, ymin

# ________________Debugging___________________________
# Imports
# import imageio
# # from process_slides import load_slides

# #Variablen
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

#tumor training 
# label_dict = {'Bg':0 , 'Tumor': 1, 'Normal': 2, 'Nekrose / Entzündung': 2, 
#               'Gefäße - Frei': 1, 'Gefäße - Metastase': 2 
#               ,'Lumen': 0,
#               'Tub. 1S - Pleo 1': 1, 'Tub. 1S - Pleo 2': 1, 'Tub. 1S - Pleo 3': 1, 
#               'Tub. PluriS - Pleo 1': 1, 'Tub. PluriS - Pleo 2': 1, 'Tub. PluriS - Pleo 3':1,
#              'Tubulopap. - Pleo 1': 1, 'Tubulopap. - Pleo 2': 1, 'Tubulopap. - Pleo 3': 1, 
#             'Solid - Pleo 1': 1, 'Solid - Pleo 2': 1, 'Solid - Pleo 3': 1,
#               'Tumorstroma': 1, 
#               'Ignore': -1, 'Annotationskasten': -1,
#              'Micropap. P1': 1, 'Micropap. P2': 1, 'Micropap. P3': 1,
             
#              'Kasten - vereinfacht': -1, 'Kasten - Test': -1, 
#               'A1 - B1': -1,'A1 - B2': -1, 'A1 - B3': -1, 'A2 - B1': -1,'A2 - B2': -1,'A2 - B3':-1, 'A3 - B1':-1, 'A3 - B2':-1,'A3 - B3':-1}

# label_dict = {'Tumor': 1, 'KeinTumor': 2, 'Other': -1}
# # # # # slide dimensions (4198, 2262)

# #Slide Container Objekt: ACHTUNG EVENTUELL DB ANPASSEN 
# sc = SlideContainer(file = Path("/home/klose/Data/crops/CP_CMC_16_954 B.svs_1.tiff"), annotation_file = "/home/klose/Data/CMC_Tumor_NonTumor.json", width= 4198, height = 2262,
# label_dict=label_dict)

# sc.get_y_patch()


# #Test image file
# file = sc.__str__()
# print(file)

# #einzelnes Patch bekommen - ohne Annotationen
# img = sc.get_patch()
# print(img.shape)


# #schwarz weiß Patch nur mit Annotationen bekommen
# img_anno = sc.get_y_patch()
# print(img_anno.shape)
# imageio.imwrite("Data/segemted_neu.jpg", img_anno)
# print(thumbnail_test.__annotations__)

# # target_folder = "/home/klose/Data/crops/CP_CMC_16_954 B.svs_1.tiff"
# # annotation_file = "/home/klose/Data/CMC.json"
# # train, valid, test = load_slides(set, patch_size=2000, label_dict=label_dict, annotation_file=annotation_file, target_folder=target_folder)
