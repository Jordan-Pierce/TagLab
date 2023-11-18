from source.Channel import Channel
from source.Blob import Blob
from source.Shape import Layer, Shape
from source.Annotation import Annotation
from source.Grid import Grid
import rasterio as rio
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)



class Image(object):
    def __init__(self, rect=[0.0, 0.0, 0.0, 0.0], map_px_to_mm_factor=1.0, width=None, height=None, channels=[],
                 id=None, name=None, acquisition_date="", georef_filename="", workspace=[], metadata={},
                 annotations={}, layers=[], grid={}, export_dataset_area=[]):

        # Notes:
        # We have to select a standard to enforce!
        # In image standard (x, y, width height)
        # In numpy standard (y, x, height, width); no the mixed format we use now I REFUSE to use it.
        # In range np format: (top, left, bottom, right)
        # In GIS standard (bottom, left, top, right)

        # Coordinates of the image (in the spatial reference system)
        self.rect = rect

        # If we have a references system we should be able to recover this number,
        # Otherwise we need to specify it.
        self.map_px_to_mm_factor = map_px_to_mm_factor

        # Dimensions in pixels
        self.width = width
        self.height = height

        self.annotations = Annotation()
        for data in annotations:
            blob = Blob(None, 0, 0, 0)
            blob.fromDict(data)
            self.annotations.addBlob(blob)

        self.layers = []
        for layer_data in layers:
            layer = Layer(layer_data["type"])
            layer.name = layer_data["name"]
            for data in layer_data["shapes"]:
                shape = Shape(None, None)
                shape.fromDict(data)
                layer.shapes.append(shape)
            self.layers.append(layer)

        self.channels = list(map(lambda c: Channel(**c), channels))

        # Internal id used in correspondences it will never change
        self.id = id
        # A label for an annotated image
        self.name = name
        # A polygon in spatial reference system (reserved for future uses)
        self.workspace = workspace
        # This is the region exported for training
        self.export_dataset_area = export_dataset_area
        # Acquisition date is mandatory (format YYYY-MM-DD)
        self.acquisition_date = acquisition_date
        # Image file (GeoTiff) contained the geo-referencing information
        self.georef_filename = georef_filename
        # This follows image_metadata_template, do we want to allow freedom to add custom values?
        self.metadata = metadata

        if grid:
            self.grid = Grid()
            self.grid.fromDict(grid)
        else:
            self.grid = None

        self.cache_data_table = None
        self.cache_labels_table = None

    def deleteLayer(self, layer):
        """

        """
        self.layers.remove(layer)

    def pixelSize(self):
        """

        """

        if self.map_px_to_mm_factor == "":
            return 1.0
        else:
            return float(self.map_px_to_mm_factor)

    def loadGeoInfo(self, filename):
        """
        Update the geo-referencing information.
        """
        img = rio.open(filename)
        if img.crs is not None:
            # This image contains geo-reference information
            self.georef_filename = filename

    def addChannel(self, filename, type):
        """
        This image add a channel to this image. The functions update the size (in pixels) and
        the geo-referencing information (if the image is geo-referenced).

        The image data is loaded when the image channel is used for the first time.
        """

        img = rio.open(filename)

        # Check image size consistency (all the channels must have the same size)
        if self.width is not None and self.height is not None:
            if self.width != img.width or self.height != img.height:
                raise Exception("Size of the images is not consistent! It is " + str(img.width) + "x" +
                                str(img.height) + ", should have been: " + str(self.width) + "x" + str(self.height))

        # Check image size limits
        if img.width > 32767 or img.height > 32767:
            raise Exception(
                "This map exceeds the image dimension handled by TagLab (the maximum size is 32767 x 32767).")

        if img.crs is not None:
            # This image contains geo-reference information
            self.georef_filename = filename

        self.width = img.width
        self.height = img.height

        self.channels.append(Channel(filename, type))

    def create_labels_table(self, labels):
        """
        Creates a data table for the label panel
        """

        if self.annotations.table_needs_update is False:
            return self.cache_labels_table
        else:
            dict = {
                'Visibility': np.zeros(len(labels), dtype=np.int32),
                'Color': [],
                'Class': [],
                '#': np.zeros(len(labels), dtype=np.int32),
                'Coverage': np.zeros(len(labels), dtype=np.float)
            }

            for i, label in enumerate(labels):
                dict['Visibility'][i] = np.int32(label.visible)
                dict['Color'].append(str(label.fill))
                dict['Class'].append(label.name)
                count, new_area = self.annotations.calculate_perclass_blobs_value(label, self.map_px_to_mm_factor)
                dict['#'][i] = count
                dict['Coverage'][i] = new_area

            # create dataframe
            df = pd.DataFrame(dict, columns=['Visibility', 'Color', 'Class', '#', 'Coverage'])
            self.cache_labels_table = df
            self.annotations.table_needs_update = False
            return df

    def create_data_table(self):
        """
        This creates a data table only for the data panel view
        """

        if self.annotations.table_needs_update is False:
            return self.cache_data_table
        else:
            scale_factor = self.pixelSize()

            # Create a list of instances
            name_list = []
            visible_blobs = []
            for blob in self.annotations.seg_blobs:
                if blob.qpath_gitem is not None:
                    if blob.qpath_gitem.isVisible():
                        index = blob.blob_name
                        name_list.append(index)
                        visible_blobs.append(blob)

            number_of_seg = len(name_list)
            dict = {
                'Id': np.zeros(number_of_seg, dtype=np.int32),
                'Class': [],
                'Area': np.zeros(number_of_seg),
                # 'Surf. area': np.zeros(number_of_seg)
            }

            for i, blob in enumerate(visible_blobs):
                dict['Id'][i] = blob.id
                dict['Class'].append(blob.class_name)
                dict['Area'][i] = round(blob.area * scale_factor * scale_factor / 100, 2)

            df = pd.DataFrame(dict, columns=['Id', 'Class', 'Area'])
            self.cache_data_table = df
            self.annotations.table_needs_update = False

            return df

    def updateChannel(self, filename, type):
        """

        """
        img = rio.open(filename)

        # Check image size consistency (all the channels must have the same size)
        if self.width is not None and self.height is not None:
            if self.width != img.width or self.height != img.height:
                raise Exception("Size of the images is not consistent! It is " + str(img.width) + "x" +
                                str(img.height) + ", should have been: " + str(self.width) + "x" + str(self.height))

        if img.crs is not None:
            # This image contains geo-reference information
            self.georef_filename = filename

        for index, channel in enumerate(self.channels):
            if channel.type == type:
                self.channels[index] = Channel(filename, type)

    def hasDEM(self):
        """
        It returns True if the image has a DEM channel, False otherwise.
        """
        for channel in self.channels:
            if channel.type == "DEM":
                return True

        return False

    def getChannel(self, type):
        """

        """
        for channel in self.channels:
            if channel.type == type:
                return channel
        return None

    def getChannelIndex(self, channel):
        """

        """
        try:
            index = self.channels.index(channel)
            return index
        except:
            return -1

    def getRGBChannel(self):
        """
        It returns the RGB channel (if exists).
        """
        return self.getChannel("RGB")

    def getDEMChannel(self):
        """
        It returns the DEM channel (if exists).
        """
        return self.getChannel("DEM")

    def save(self):
        data = self.__dict__.copy()

        # cached tables MUST NOT be saved
        del data["cache_data_table"]
        del data["cache_labels_table"]

        return data