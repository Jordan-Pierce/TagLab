from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPen, QBrush

from source.Blob import Blob
from source.tools.Tool import Tool
from source.utils import qimageToNumpyArray
from source.utils import cropQImage

import os
import cv2
import numpy as np
from skimage.measure import regionprops

import torch

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

from models.dataloaders import helpers as helpers


class SAMGenerator(Tool):

    def __init__(self, viewerplus, corrective_points):
        super(SAMGenerator, self).__init__(viewerplus)

        # Image is resized to
        self.resize_to = 1024
        # Padding amount
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_b'
        # Mask score threshold
        self.score_threshold = 0.80
        # For debugging
        self.debug = False

        # SAM, CUDA or CPU
        self.samgenerator_net = None
        self.device = None

        # Drawing on GUI
        self.CROSS_LINE_WIDTH = 2
        self.pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.red, 'size': 6}
        self.work_area_bbox = None
        self.work_area_item = None

    def leftPressed(self, x, y, mods):
        """

        """

        # Load Network in the beginning
        self.loadNetwork()

        # If the weights are there, continue
        if self.samgenerator_net:

            # User left-clicks without shift, working area is set
            # They can then zoom out if they want to adjust
            if mods != Qt.ShiftModifier:
                # Update working area
                self.getWorkArea()

            # Single Click
            if mods == Qt.ShiftModifier:
                message = "[TOOL][SAMGENERATOR] Segmentation activated"
                self.log.emit(message)
                # Segment with SAM
                self.segmentWithSAMGenerator()


    def getWorkArea(self):
        """
        User defined work area
        """
        self.resetWorkArea()

        rect_map = self.viewerplus.viewportToScene()
        self.work_area_bbox = [round(rect_map.top()),
                               round(rect_map.left()),
                               round(rect_map.width()),
                               round(rect_map.height())]

        # Display to GUI
        brush = QBrush(Qt.NoBrush)
        pen = QPen(Qt.DashLine)
        pen.setWidth(2)
        pen.setColor(Qt.white)
        pen.setCosmetic(True)
        x = self.work_area_bbox[1]
        y = self.work_area_bbox[0]
        w = self.work_area_bbox[2]
        h = self.work_area_bbox[3]
        self.work_area_item = self.viewerplus.scene.addRect(x, y, w, h, pen, brush)
        self.work_area_item.setZValue(3)

    def prepareForSAMGenerator(self):
        """
        Obtain the image from defined work area
        """

        image_cropped = cropQImage(self.viewerplus.img_map, self.work_area_bbox)
        image_cropped = qimageToNumpyArray(image_cropped)

        return image_cropped

    def segmentWithSAMGenerator(self):

        if not self.viewerplus.img_map:
            return

        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMGENERATOR] Segmentation begins..")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Get the work area top-left
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]

        # Crop the image from the work area
        image_cropped = self.prepareForSAMGenerator()
        # Resize the cropped image
        image_resized = helpers.fixed_resize(image_cropped,
                                             (self.resize_to, self.resize_to)).astype(np.uint8)

        # Generate masks
        generated_outputs = self.samgenerator_net.generate(image_resized)

        for idx, generated_output in enumerate(generated_outputs):

            # Extract the output
            mask_resized = generated_output['segmentation']
            area = generated_output['area']
            stable_score = generated_output['stability_score']
            iou_score = generated_output['predicted_iou']
            crop_box = generated_output['crop_box']
            x, y, w, h = generated_output['bbox']
            x2, y2 = x + w, y + h

            # Remove masks that form at the boundaries of images?
            # May remove this if users prefer to keep those and clean
            # manually.
            eps = 5

            if 0 in [x, y] or x2 >= mask_resized.shape[1] - eps or y2 >= mask_resized.shape[0] - eps:
                continue

            # Resize mask back to original cropped size
            mask_cropped = cv2.resize(mask_resized.astype(float),
                                      (image_cropped.shape[1], image_cropped.shape[0]),
                                      cv2.INTER_NEAREST).astype(int)

            # Calculate scale
            x_scale = mask_cropped.shape[1] / mask_resized.shape[1]
            y_scale = mask_cropped.shape[0] / mask_resized.shape[0]

            try:
                # Create region manually since information is available;
                # It's also much faster than using scikit measure...

                # Inside a try block because scikit complains, but still
                # takes the values anyways
                region = regionprops(mask_cropped)[0]
                region.label = 1
                region.bbox = (x * x_scale,
                               y * y_scale,
                               (x + w) * x_scale,
                               (y + h) * y_scale)
                region.area = np.sum(mask_cropped)
                region.centroid = np.mean(np.argwhere(mask_cropped), axis=0)
            except:
                pass

            blob_id = self.viewerplus.annotations.getFreeId()
            blob = Blob(region, left_map_pos, top_map_pos, blob_id)

            # Exclude all masks outside of mosaic
            if np.all(blob.bbox >= 0):
                self.viewerplus.addBlob(blob, selected=True)
                self.blobInfo.emit(blob, "[TOOL][SAMGENERATOR][BLOB-CREATED]")
            self.viewerplus.saveUndo()

        QApplication.restoreOverrideCursor()
        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMGENERATOR] Segmentation ends.")
        self.resetWorkArea()

    def loadNetwork(self):

        if self.samgenerator_net is None:
            self.infoMessage.emit("Loading SAM network..")

            # Mapping between the model type, and the checkpoint file name
            sam_dict = {"vit_b": "sam_vit_b_01ec64",
                        "vit_l": "sam_vit_l_0b3195",
                        "vit_h": "sam_vit_h_4b8939"}

            # Initialization
            modelName = sam_dict[self.sam_model_type]
            models_dir = os.path.join(self.viewerplus.taglab_dir, "models")
            path = os.path.join(models_dir, modelName + '.pth')

            if not os.path.exists(path):

                # Create a box with a warning
                box = QMessageBox()
                box.setText(f"Model weights {self.sam_model_type} cannot be found in models folder.\n"
                            f"If they have not been downloaded, re-run the install script.")
                box.exec()
                # Go back to GUI without closing program

            else:
                # Set the device; users should be using a CUDA GPU, otherwise tool is slow
                device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

                # Loading the model, returning the predictor
                sam_model = sam_model_registry[self.sam_model_type](checkpoint=path)
                sam_model.to(device=device)
                self.samgenerator_net = SamAutomaticMaskGenerator(sam_model,
                                                                  points_per_side=32,
                                                                  points_per_batch=128)
                self.device = device

    def resetNetwork(self):
        """
        Reset the network
        """

        torch.cuda.empty_cache()
        if self.samgenerator_net is not None:
            del self.samgenerator_net
            self.samgenerator_net = None

    def resetWorkArea(self):
        """
        Reset working area
        """
        self.work_area_bbox = [0, 0, 0, 0]
        if self.work_area_item is not None:
            self.viewerplus.scene.removeItem(self.work_area_item)
            self.work_area_item = None

    def reset(self):
        """
        Reset everything
        """
        self.resetNetwork()
        self.resetWorkArea()
