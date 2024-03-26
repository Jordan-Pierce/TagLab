from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPen, QBrush, QFont

from source.Blob import Blob
from source.tools.Tool import Tool
from source.genutils import qimageToNumpyArray, cropQImage

import os
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

import torch

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from models.dataloaders import helpers as helpers


class SAMPredictor(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMPredictor, self).__init__(viewerplus)
        # User defined points
        self.pick_points = pick_points

        # Image is resized to
        self.longest_side = 2048
        # Resize dimensions
        self.resize_width = None
        self.resize_height = None
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_b'
        # Mask score threshold
        self.score_threshold = 0.70
        # Labels for fore/background
        self.labels = []
        # For debugging
        self.debug = False

        # Mosaic dimensions
        self.width = None
        self.height = None

        # Set image
        self.image_resized = None
        self.image_cropped = None

        # SAM, CUDA or CPU
        self.sampredictor_net = None
        self.device = None

        # Drawing on GUI
        self.work_area_bbox = None
        self.work_area_item = None
        self.work_area_text_tl = None
        self.work_area_text_br = None

        # Updating masks/blobs
        self.current_blobs = []
        self.blob_to_correct = None

        self.CROSS_LINE_WIDTH = 2
        self.work_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.cyan, 'size': 8}
        self.pos_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.green, 'size': 6}
        self.neg_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.red, 'size': 6}

    def leftPressed(self, x, y, mods):
        """
        Positive points
        """

        self.loadNetwork()

        # User is still selecting work area
        if not self.sampredictor_net.is_image_set:

            # Collect points to set work area
            if len(self.pick_points.points) < 2:
                self.pick_points.addPoint(x, y, self.work_pick_style)

            # User choose a third point without pressing
            # SPACE, so reset the work area points, add
            else:
                self.pick_points.reset()
                self.pick_points.addPoint(x, y, self.work_pick_style)

        # User has already selected a work area, and now
        # is choosing positive or negative points
        else:

            # Add points
            self.pick_points.addPoint(x, y, self.pos_pick_style)
            self.labels.append(1)
            message = "[TOOL][SAMPREDICTOR] New point picked"
            self.log.emit(message)
            # Segment with current points
            if self.points_within_workarea():
                self.segmentWithSAMPredictor()

    def rightPressed(self, x, y, mods):
        """
        Negative points
        """

        self.loadNetwork()

        if mods == Qt.ShiftModifier:

            # User is still selecting work area
            if not self.sampredictor_net.is_image_set:

                # Collect points to set work area
                if len(self.pick_points.points) < 2:
                    self.pick_points.addPoint(x, y, self.work_pick_style)

                # User choose a third point without pressing
                # SPACE, so reset the work area points, add
                else:
                    self.pick_points.reset()
                    self.pick_points.addPoint(x, y, self.work_pick_style)

            # User has already selected a work area, and now
            # is choosing positive or negative points
            else:
                # Add points
                self.pick_points.addPoint(x, y, self.neg_pick_style)
                self.labels.append(0)
                message = "[TOOL][SAMPREDICTOR] New point picked"
                self.log.emit(message)
                # Segment with current points
                if self.points_within_workarea():
                    self.segmentWithSAMPredictor()

    def apply(self):
        """
        User presses SPACE to set work area, and again later to run the model
        """
        # User has chosen the current view as the working area, saving work area
        if len(self.pick_points.points) == 0 and self.sampredictor_net is None:
            self.loadNetwork()
            self.getExtent()
            self.setWorkArea()

        # User has finished selecting points, submitting current blob
        elif len(self.pick_points.points) and self.sampredictor_net.is_image_set:
            self.submitBlobs()

        # User has finished creating working area, saving work area
        elif len(self.pick_points.points) == 2 and not self.sampredictor_net.is_image_set:
            self.setWorkArea()

    def points_within_workarea(self):
        """
        Checks if selected points are within established work area
        """
        # Define the boundaries
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]
        width_map_pos = self.work_area_bbox[2]
        height_map_pos = self.work_area_bbox[3]

        # Check if any points are outside the boundaries
        points = np.array(self.pick_points.points)
        outside_boundaries = (
                (points[:, 0] < left_map_pos) |
                (points[:, 0] > left_map_pos + width_map_pos) |
                (points[:, 1] < top_map_pos) |
                (points[:, 1] > top_map_pos + height_map_pos)
        )

        return not np.any(outside_boundaries)

    def getExtent(self):
        """

        """
        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        # Current extent
        rect_map = self.viewerplus.viewportToScene()

        top = round(rect_map.top())
        left = round(rect_map.left())
        width = round(rect_map.width())
        height = round(rect_map.height())
        bottom = top + height
        right = left + width

        # If the current extent includes areas outside the
        # mosaic, reduce it to be only the mosaic
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > self.height:
            bottom = self.height
        if right > self.width:
            right = self.width

        self.pick_points.addPoint(left, top, self.work_pick_style)
        self.pick_points.addPoint(left, bottom, self.work_pick_style)
        self.pick_points.addPoint(right, bottom, self.work_pick_style)
        self.pick_points.addPoint(right, top, self.work_pick_style)

    def setAspectRatio(self):
        """

        """
        # Calculate the original width and height
        original_height, original_width = self.image_cropped.shape[0:2]

        # Calculate the new height to preserve the aspect ratio
        new_height = int((self.longest_side / original_width) * original_height)

        self.resize_width = self.longest_side
        self.resize_height = new_height

    def setWorkArea(self):
        """
        Set the work area based on the location of points
        """

        from source.QtImageViewerPlus import TextItem

        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        points = np.array(self.pick_points.points)

        x = points[:, 0].min()
        y = points[:, 1].min()
        w = points[:, 0].max() - x
        h = points[:, 1].max() - y

        self.work_area_bbox = [round(y), round(x), round(w), round(h)]

        # Draw box
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

        # Draw box dimensions (tl)
        font_size = 8
        text_item = TextItem(f"{w} x {h}", QFont("Roboto", font_size, QFont.Bold))
        text_item.setPos(x + (len(str(w)) + 1) * font_size, y - font_size)
        text_item.setTransformOriginPoint(QPointF(x, y))
        text_item.setZValue(3)
        text_item.setBrush(Qt.white)
        text_item.setOpacity(1.0)
        self.work_area_text_tl = text_item
        self.viewerplus.scene.addItem(self.work_area_text_tl)

        # Draw box dimensions (br)
        font_size = 8
        text_item = TextItem(f"{w} x {h}", QFont("Roboto", font_size, QFont.Bold))
        text_item.setPos((x+w) - (len(str(w)) + 1) * font_size, (y+h) + font_size)
        text_item.setTransformOriginPoint(QPointF(x+w, y+h))
        text_item.setZValue(3)
        text_item.setBrush(Qt.white)
        text_item.setOpacity(1.0)
        self.work_area_text_br = text_item
        self.viewerplus.scene.addItem(self.work_area_text_br)

        # From the current view, crop the image
        image_cropped = cropQImage(self.viewerplus.img_map, self.work_area_bbox)
        self.image_cropped = qimageToNumpyArray(image_cropped)

        # Set the aspect ratio of the cropped image
        self.setAspectRatio()

        # Resize the cropped QImage
        self.image_resized = helpers.fixed_resize(self.image_cropped,
                                                  (self.resize_width, self.resize_height)).astype(np.uint8)

        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Don't use torch here, as the image resize is fixed to 1024
        self.sampredictor_net.set_image(self.image_resized)
        self.pick_points.reset()

        QApplication.restoreOverrideCursor()

    def resizeArray(self, arr, shape, interpolation=cv2.INTER_CUBIC):
        """
        Resize array; expects 2D array.
        """
        return cv2.resize(arr.astype(float), shape, interpolation)

    def preparePoints(self):
        """
        Get the image based on point(s) location
        """
        points = np.asarray(self.pick_points.points).astype(int)

        left = self.work_area_bbox[1]
        top = self.work_area_bbox[0]

        # Update points to be in image_cropped coordinate space
        points_cropped = np.zeros((len(points), 2), dtype=np.int32)
        points_cropped[:, 0] = points[:, 0] - left
        points_cropped[:, 1] = points[:, 1] - top

        # Points in the resized image
        x_scale = self.image_resized.shape[1] / self.image_cropped.shape[1]
        y_scale = self.image_resized.shape[0] / self.image_cropped.shape[0]

        # New coordinates
        points_resized = np.zeros_like(points_cropped, dtype=np.float32)
        points_resized[:, 0] = points_cropped[:, 0] * x_scale
        points_resized[:, 1] = points_cropped[:, 1] * y_scale

        return points_cropped, points_resized

    def segmentWithSAMPredictor(self):
        """

        """
        if not self.viewerplus.img_map:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation begins..")

        # Get the work area top-left
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]

        # Points in the cropped image
        points_cropped, points_resized = self.preparePoints()

        # Convert to torch, cuda
        input_labels = torch.tensor(np.array(self.labels)).to(self.device).unsqueeze(0)
        input_points = torch.as_tensor(points_resized.astype(int), dtype=torch.int64).to(self.device).unsqueeze(0)
        transformed_points = self.sampredictor_net.transform.apply_coords_torch(input_points,
                                                                                self.image_resized.shape[:2])

        # Make prediction given points
        mask, score, logit = self.sampredictor_net.predict_torch(point_coords=transformed_points,
                                                                 point_labels=input_labels,
                                                                 multimask_output=False)

        # Move back to CPU
        mask = mask.detach().cpu().numpy()
        score = score.detach().cpu().numpy()

        # If mask score is too low, just return early
        if score.squeeze() < self.score_threshold:
            self.infoMessage.emit("Predicted mask score is too low, skipping...")
        else:
            # Get the mask as a float
            mask_resized = mask.squeeze()
            # Fill in while still small
            mask_resized = ndi.binary_fill_holes(mask_resized).astype(float)

            # Region contain masked object
            indices = np.argwhere(mask_resized)

            # Calculate the x, y, width, and height
            x = indices[:, 1].min()
            y = indices[:, 0].min()
            w = indices[:, 1].max() - x + 1
            h = indices[:, 0].max() - y + 1
            bbox = np.array([x, y, w, h])

            # Resize mask back to cropped size
            target_shape = (self.image_cropped.shape[:2][::-1])
            mask_cropped = self.resizeArray(mask_resized, target_shape, cv2.INTER_LINEAR).astype(np.uint8)

            self.undrawAllBlobs()

            # Create a blob manually using provided information
            blob = self.createBlob(mask_resized, mask_cropped, bbox, left_map_pos, top_map_pos)

            if blob:
                self.current_blobs.append(blob)
                self.drawBlob(blob)

        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation ends.")

        QApplication.restoreOverrideCursor()

    def smoothVertices(self, arr, window_size=3):
        """

        """
        # Find the contours in the binary mask
        contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose the largest contour if there are multiple
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices from the largest contour
        vertices = [tuple(point[0]) for point in largest_contour]

        # Smooth the vertices
        smoothed_vertices = []
        half_window = window_size // 2

        for i in range(len(vertices)):
            start = max(0, i - half_window)
            end = min(len(vertices), i + half_window + 1)
            avg_x = sum(p[0] for p in vertices[start:end]) / (end - start)
            avg_y = sum(p[1] for p in vertices[start:end]) / (end - start)
            smoothed_vertices.append((avg_x, avg_y))

        # Convert smoothed vertices back to numpy array
        smoothed_contour = np.array(smoothed_vertices, dtype=np.int32)

        # Create a new binary mask with the smoothed contour
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        cv2.fillPoly(new_arr, [smoothed_contour], 1)

        return new_arr

    def createBlob(self, mask_src, mask_dst, bbox_src, left_map_pos, top_map_pos):
        """
        Create a blob manually given the generated mask
        """

        # Bbox of the area of interest before scaled
        x1_src, y1_src, w_src, h_src = bbox_src

        # Calculate scale
        x_scale = mask_dst.shape[1] / mask_src.shape[1]
        y_scale = mask_dst.shape[0] / mask_src.shape[0]

        # New coordinates
        x1_dst = x1_src * x_scale
        y1_dst = y1_src * y_scale
        w_dst = w_src * x_scale
        h_dst = h_src * y_scale

        # Bbox of the area of interest after scaled
        bbox_dst = (x1_dst, y1_dst, (x1_dst + w_dst), (y1_dst + h_dst))

        try:
            # Create region manually since information is available;
            # It's also much faster than using scikit measure

            # Inside a try block because scikit complains, but still
            # takes the values anyway
            region = sorted(regionprops(mask_dst), key=lambda r: r.area, reverse=True)[0]
            region.label = 1
            region.bbox = bbox_dst
            region.area = np.sum(mask_dst)
            region.centroid = np.mean(np.argwhere(mask_dst), axis=0)
        except:
            pass

        blob_id = self.viewerplus.annotations.getFreeId()
        blob = Blob(region, left_map_pos, top_map_pos, blob_id)

        return blob

    def drawBlob(self, blob):
        """

        """
        # get the scene
        scene = self.viewerplus.scene

        # if it has just been created remove the current graphics item in order to set it again
        if blob.qpath_gitem is not None:
            scene.removeItem(blob.qpath_gitem)
            del blob.qpath_gitem
            blob.qpath_gitem = None

        blob.setupForDrawing()

        pen = QPen(Qt.white)
        pen.setWidth(2)
        pen.setCosmetic(True)

        if self.blob_to_correct is None:
            brush = QBrush(Qt.SolidPattern)
            brush.setColor(Qt.white)
        else:
            brush = self.viewerplus.project.classBrushFromName(self.blob_to_correct)

        brush.setStyle(Qt.Dense4Pattern)

        blob.qpath_gitem = scene.addPath(blob.qpath, pen, brush)
        blob.qpath_gitem.setZValue(1)
        blob.qpath_gitem.setOpacity(self.viewerplus.transparency_value)

    def undrawBlob(self, blob):
        """

        """
        # Get the scene
        scene = self.viewerplus.scene
        # Undraw
        scene.removeItem(blob.qpath_gitem)
        blob.qpath = None
        blob.qpath_gitem = None
        scene.invalidate()

    def undrawAllBlobs(self):
        """

        """
        # Undraw all blobs in list
        if len(self.current_blobs) > 0:
            for blob in self.current_blobs:
                self.undrawBlob(blob)
        self.current_blobs = []

    def submitBlobs(self):
        """

        """
        # Finalize created blob
        message = "[TOOL][SAMPREDICTOR][BLOB-CREATED]"
        for blob in self.current_blobs:

            if self.blob_to_correct is not None:
                self.viewerplus.removeBlob(self.blob_to_correct)
                blob.id = self.blob_to_correct.id
                blob.class_name = self.blob_to_correct.class_name
                message = "[TOOL][SAMPREDICTOR][BLOB-EDITED]"

            # order is important: first add then setblob class!
            self.undrawBlob(blob)
            self.viewerplus.addBlob(blob, selected=True)
            self.blobInfo.emit(blob, message)

        self.viewerplus.saveUndo()
        self.viewerplus.resetSelection()
        self.pick_points.reset()
        self.labels = []
        self.current_blobs = []
        self.blob_to_correct = None

    def loadNetwork(self):

        if self.sampredictor_net is None:

            QApplication.setOverrideCursor(Qt.WaitCursor)
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
                self.sampredictor_net = SamPredictor(sam_model)
                self.device = device

            QApplication.restoreOverrideCursor()

    def resetNetwork(self):
        """
        Reset the network
        """
        torch.cuda.empty_cache()
        if self.sampredictor_net is not None:
            del self.sampredictor_net
            self.sampredictor_net = None

    def resetWorkArea(self):
        """
        Reset working area
        """
        self.image_resized = None
        self.image_cropped = None
        self.work_area_bbox = [0, 0, 0, 0]
        if self.work_area_item is not None:
            self.viewerplus.scene.removeItem(self.work_area_text_tl)
            self.viewerplus.scene.removeItem(self.work_area_text_br)
            self.viewerplus.scene.removeItem(self.work_area_item)
            self.work_area_text_tl = None
            self.work_area_text_br = None
            self.work_area_item = None

    def reset(self):
        """
        Reset everything
        """
        self.resetNetwork()
        self.undrawAllBlobs()
        self.pick_points.reset()
        self.labels = []
        self.resetWorkArea()