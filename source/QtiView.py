# TagLab                                               
# A semi-automatic segmentation tool                                    
#
# Copyright(C) 2019                                         
# Visual Computing Lab                                           
# ISTI - Italian National Research Council                              
# All rights reserved.                                                      

# This program is free software; you can redistribute it and/or modify      
# it under the terms of the GNU General Public License as published by      
# the Free Software Foundation; either version 2 of the License, or         
# (at your option) any later version.                                       

# This program is distributed in the hope that it will be useful,           
# but WITHOUT ANY WARRANTY; without even the implied warranty of            
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             
# GNU General Public License (http://www.gnu.org/licenses/gpl.txt)
# for more details.                                               

import os
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QSize, pyqtSlot, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QTransform, QFont, QImageReader, QPainter, QColor
from PyQt5.QtWidgets import QStackedWidget, QTabWidget, QScrollArea
from PyQt5.QtWidgets import QSizePolicy, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGroupBox, QWidget, QFileDialog, QComboBox, QApplication, QMessageBox
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsSceneWheelEvent

from source.QtProgressBarCustom import QtProgressBarCustom

import numpy as np
import Metashape

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version,
                                                                      compatible_major_version))


class ThumbnailWidget(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, file_path, thumbnail):
        super(ThumbnailWidget, self).__init__()

        # Get the basename of the file
        self.file_path = file_path
        self.basename = os.path.basename(file_path)

        # Create a QLabel to display the thumbnail
        self.thumbnail_label = QLabel(self)
        self.thumbnail_label.setPixmap(thumbnail)

        # Create a QLabel to display the basename
        self.basename_label = QLabel(self.basename, self)
        font = QFont()
        font.setPointSize(6)
        self.basename_label.setFont(font)
        self.basename_label.setAlignment(Qt.AlignCenter)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.thumbnail_label)
        layout.addWidget(self.basename_label)
        self.setLayout(layout)

        # For the red highlight indicating selection
        # Initialize the selected state to False
        self.selected = False

    def mousePressEvent(self, event):
        # Emit the clicked signal with the file_path when the widget is clicked
        self.clicked.emit(self.file_path)
        # Update the visual state based on the selected state
        self.setSelected(not self.selected)
        super().mousePressEvent(event)

    def setSelected(self, selected):
        # Only update the visual state if the selection status is changing
        if self.selected != selected:
            # Update the selected state
            self.selected = selected


def create_thumbnail(file_path):
    """
    Used for parallelism, much faster.
    """
    # Reads the image path, converts to RGB
    image_reader = QImageReader(file_path)
    image_reader.setAutoTransform(True)

    # Read the image using QImage
    image = image_reader.read()

    # Display the image using a QPixmap
    pixmap = QPixmap.fromImage(image)
    thumbnail = pixmap.scaled(90, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return file_path, thumbnail


class QtiView(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super(QtiView, self).__init__(parent)

        # Metashape
        self.metashapeLicense = False
        self.metashapeProject = None
        self.metashapeChunk = None
        self.metashapeChunks = []
        self.metashapeOrthomosaic = None
        self.metashapeOrthomosaics = []
        self.closestImages = []
        self.thumbnailWidgets = {}

        # --------------------
        # The window settings
        # --------------------
        self.setWindowTitle("iView")

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # Set size policy to allow resizing
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        self.setStyleSheet("background-color: rgba(60,60,65,100); color: white")

        # Create a button to toggle the visibility of the panel
        self.toggle_button = QPushButton('Toggle Parameters', self)
        self.toggle_button.clicked.connect(self.togglePanel)
        button_stylesheet = "QPushButton { background-color: rgb(150, 150, 150); }"
        self.toggle_button.setStyleSheet(button_stylesheet)

        # Create a stacked widget to manage the visibility of the tabbed panels
        self.stacked_widget = QStackedWidget(self)

        # Create a tab widget for the tabbed panels
        self.tabWidget = QTabWidget()
        self.tabWidget.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        # ---------------------
        # Metashape Panel
        # ---------------------

        # Metashape Tab
        metashapeTab = QWidget()
        metashapeTab.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        # Metashape Panel, within Tab
        layoutMetashapePanel = QVBoxLayout(metashapeTab)

        # Metashape License Field
        layoutLicense = QHBoxLayout()
        layoutLicense.setAlignment(Qt.AlignLeft)
        self.lblLicense = QLabel("License: ")
        self.lblLicense.setFixedWidth(130)
        self.lblLicense.setMinimumWidth(130)
        self.editLicense = QLineEdit("")
        self.editLicense.setEchoMode(QLineEdit.Password)
        self.editLicense.setPlaceholderText("Metashape License")
        self.editLicense.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnLicense = QPushButton("...")
        self.btnLicense.setFixedWidth(20)
        self.btnLicense.setMinimumWidth(20)
        self.btnLicense.clicked.connect(self.getMetashapeLicense)
        layoutLicense.addWidget(self.lblLicense)
        layoutLicense.addWidget(self.editLicense)
        layoutLicense.addWidget(self.btnLicense)

        # Metashape Project Field
        layoutProject = QHBoxLayout()
        layoutProject.setAlignment(Qt.AlignLeft)
        self.lblProject = QLabel("Project: ")
        self.lblProject.setFixedWidth(130)
        self.lblProject.setMinimumWidth(130)
        self.editProject = QLineEdit("")
        self.editProject.setPlaceholderText("Path to Metashape Project")
        self.editProject.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnProject = QPushButton("...")
        self.btnProject.setFixedWidth(20)
        self.btnProject.setMinimumWidth(20)
        self.btnProject.clicked.connect(self.chooseMetashapeProject)
        layoutProject.addWidget(self.lblProject)
        layoutProject.addWidget(self.editProject)
        layoutProject.addWidget(self.btnProject)

        # Metashape Load (button)
        layoutLoadMetashape = QHBoxLayout()
        layoutLoadMetashape.setAlignment(Qt.AlignLeft)
        button_stylesheet = "QPushButton { background-color: rgb(150, 150, 150); }"
        self.btnLoadMetashape = QPushButton("Load Project")
        self.btnLoadMetashape.clicked.connect(self.loadMetashape)
        self.btnLoadMetashape.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btnLoadMetashape.setStyleSheet(button_stylesheet)
        self.btnLoadMetashape.setMinimumWidth(500)
        layoutLoadMetashape.addWidget(self.btnLoadMetashape)

        # Metashape Chunk (combobox)
        layoutChunk = QHBoxLayout()
        layoutChunk.setAlignment(Qt.AlignLeft)
        self.lblChunk = QLabel("Chunk: ")
        self.lblChunk.setFixedWidth(130)
        self.lblChunk.setMinimumWidth(130)
        self.comboChunk = QComboBox()
        self.comboChunk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.comboChunk.setMinimumWidth(200)
        layoutChunk.addWidget(self.lblChunk)
        layoutChunk.addWidget(self.comboChunk)

        # Metashape Orthomosaic (combobox)
        layoutOrthomosaic = QHBoxLayout()
        layoutOrthomosaic.setAlignment(Qt.AlignLeft)
        self.lblOrthomosaic = QLabel("Orthomosaic: ")
        self.lblOrthomosaic.setFixedWidth(130)
        self.lblOrthomosaic.setMinimumWidth(130)
        self.comboOrthomosaic = QComboBox()
        self.comboOrthomosaic.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.comboOrthomosaic.setMinimumWidth(200)
        layoutOrthomosaic.addWidget(self.lblOrthomosaic)
        layoutOrthomosaic.addWidget(self.comboOrthomosaic)

        # Specifying the layout order
        layoutMetashapePanel.addLayout(layoutLicense)
        layoutMetashapePanel.addLayout(layoutProject)
        layoutMetashapePanel.addLayout(layoutLoadMetashape)
        layoutMetashapePanel.addLayout(layoutChunk)
        layoutMetashapePanel.addLayout(layoutOrthomosaic)

        # Add the Metashape layout to Metashape tab
        self.tabWidget.addTab(metashapeTab, "Metashape")
        self.tabWidget.tabBar().setTabTextColor(0, QColor('black'))

        # ---------------------
        # VISCORE Panel
        # ---------------------

        # VISCORE Tab
        viscoreTab = QWidget()
        viscoreTab.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutViscorePanel = QVBoxLayout(viscoreTab)

        # CODE ...

        # Add the VISCORE layout to VISCORE Tab
        self.tabWidget.addTab(viscoreTab, "VISCORE")
        self.tabWidget.tabBar().setTabTextColor(1, QColor('black'))

        # Add all tabbed panels to the stacked widget
        self.stacked_widget.addWidget(self.tabWidget)

        # ---------------------
        # iView panel
        # ---------------------
        layoutiView = QHBoxLayout()

        # ----------
        # Thumbnails
        # ----------

        # Create a QWidget to contain the thumbnails
        self.thumbnail_container = QWidget()
        self.thumbnail_container_layout = QVBoxLayout()

        # Set the widget to the scroll area
        self.scrollArea = QScrollArea()
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setFixedWidth(150)

        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.thumbnail_container)

        # Set the vertical scrollbar to the far-right
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

        # ------
        # Scene
        # ------

        # Preview image of the current image
        self.iViewWidth = 3000
        self.iViewHeight = 1980

        # Create a QGraphicsView and a QGraphicsScene
        self.graphicsView = QGraphicsView(self)
        self.scene = QGraphicsScene(self)

        # Create a blank image (black)
        blank_image = QImage(self.iViewWidth, self.iViewHeight, QImage.Format_RGB32)
        blank_image.fill(Qt.black)
        pixmap = QPixmap.fromImage(blank_image)
        # Add the image to the scene
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.graphicsView.setScene(self.scene)

        # ------------
        # Interactions
        # ------------

        # Enable zooming with the mouse wheel
        self.graphicsView.setRenderHint(QPainter.Antialiasing, False)
        self.graphicsView.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.wheelEvent = self.zoom_wheel_event

        # Enable panning and rotation with mouse press, move, and release events
        self.pan_active = False
        self.pan_start = QPoint()

        self.rotation_active = False
        self.rotation_start = QPoint()
        self.rotation_angle = 0.0

        self.graphicsView.mousePressEvent = self.mouse_press_event
        self.graphicsView.mouseMoveEvent = self.mouse_move_event
        self.graphicsView.mouseReleaseEvent = self.mouse_release_event

        # Store the original center point for rotation
        self.original_center = QPoint(self.iViewWidth / 2, self.iViewHeight / 2)

        # Initial zoom factor
        self.zoom_factor = 1.0

        # Set up the layout
        layoutClosestPreview = QVBoxLayout()
        layoutClosestPreview.setAlignment(Qt.AlignCenter)
        layoutClosestPreview.addWidget(self.graphicsView)

        # Add the layout of thumbnails and scene to the iView layout
        layoutiView.addLayout(layoutClosestPreview)
        layoutiView.addWidget(self.scrollArea)

        # Grouping the combobox and tile
        self.iViewPanel = QGroupBox("Closest Images")
        self.iViewPanel.setLayout(layoutiView)

        # -----------------------
        # Close Button
        # -----------------------
        layoutClose = QHBoxLayout()
        layoutClose.setAlignment(Qt.AlignRight)
        layoutClose.addStretch()

        self.btnExit = QPushButton("Exit")
        self.btnExit.clicked.connect(self.close)
        layoutClose.addWidget(self.btnExit)

        # -----------------------
        # Final Layout order
        # -----------------------
        layoutV = QVBoxLayout()

        # Metashape Pro, Standard, and VISCORE Tabbed panels (inside stacked)
        layoutV.addWidget(self.toggle_button)
        layoutV.addWidget(self.stacked_widget)
        layoutV.addSpacing(10)

        # iView Panel
        layoutV.addWidget(self.iViewPanel)

        # Close Panel (button)
        layoutV.addLayout(layoutClose)
        layoutV.setSpacing(3)

        self.setLayout(layoutV)

    def togglePanel(self):
        # Toggle the visibility of the stacked widget
        self.stacked_widget.setVisible(not self.stacked_widget.isVisible())


    @pyqtSlot()
    def getMetashapeLicense(self):
        """
        Automatically sets the license if previously entered
        """

        metashape_license = os.getenv("METASHAPE_LICENSE")
        if metashape_license:
            self.editLicense.setText(metashape_license)

    @pyqtSlot()
    def chooseMetashapeProject(self):
        """
        Allows thw user to navigate to the .psx file
        """

        filters = "Project File (*.psx)"
        fileName, _ = QFileDialog.getOpenFileName(self, "Input Metashape Project File", "", filters)
        if fileName:
            self.editProject.setText(fileName)

    @pyqtSlot()
    def loadMetashape(self):
        """
        Activates metashape using the license, and loads the provided .psx file
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Reset iView in case the user is loading another project
        self.resetiView()

        # Good or Bad box
        msgBox = QMessageBox()

        try:
            # Get the metashape license, try to activate
            metashape_license = self.editLicense.text()

            assert metashape_license != ""

            # Set the metashape license as environmental variable
            Metashape.License().activate(metashape_license)
            os.environ['METASHAPE_LICENSE'] = metashape_license
            self.metashapeLicense = True

            # Show a good box
            msgBox.setText("Successfully Activated Metashape!")
            msgBox.exec()

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText("Failed to Activate Metashape!")
            msgBox.exec()
            return

        try:
            # Metashape has been activated, opening document
            metashape_project = self.editProject.text()

            self.metashapeProject = Metashape.Document()
            self.metashapeProject.open(metashape_project)

            # Show a good box
            msgBox.setText("Successfully loaded Project!")
            msgBox.exec()

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText("Failed to load Project!")
            msgBox.exec()
            return

        try:
            # Project has been opened, load the chunks
            self.loadChunkComboBox()

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText(f"{e}")
            msgBox.exec()
            return

        try:
            # Chunk has been opened, load the orthomosaic
            self.loadOrthomosaicComboBox()

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText(f"{e}")
            msgBox.exec()
            return

        try:
            # Chunk has been opened, load the thumbnails
            self.loadThumbnails()

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText(f"{e}")
            msgBox.exec()
            return

        QApplication.restoreOverrideCursor()

    def loadChunkComboBox(self):
        """
        This only runs when a project is loaded (button pressed)
        """

        # Stores the chunks in global variable
        self.metashapeChunks = self.metashapeProject.chunks

        if len(self.metashapeChunks) == 0:
            raise Exception("No Chunks found in Project")

        # Remove items from existing comboChunk
        self.comboChunk.clear()

        # Add each chunk name to the combobox
        for chunk in self.metashapeChunks:
            self.comboChunk.addItem(chunk.label)

        # Set the chunk to first as default, allow updates
        self.metashapeChunk = self.metashapeChunks[0]
        self.comboChunk.currentIndexChanged.connect(self.metashapeChunkChanged)

    @pyqtSlot(int)
    def metashapeChunkChanged(self, index):
        """
        This runs everytime the selection in the combobox changes
        """
        # Reset iView as the user is loading another chunk
        self.resetiView()

        try:
            # Updates the combobox selection
            if len(self.metashapeChunks) != 0:
                # Updates the global variable
                self.metashapeChunk = self.metashapeChunks[index]
                # Updates what is seen in combobox
                self.loadOrthomosaicComboBox()
                # Updates thumbnails based on images in new chunk
                self.loadThumbnails()

        except Exception as e:
            # Show a bad box
            msgBox = QMessageBox()
            QApplication.restoreOverrideCursor()
            msgBox.setText(f"{e}")
            msgBox.exec()
            return

    def loadOrthomosaicComboBox(self):
        """
        This only runs when a project is loaded (button pressed), or when the user
        selects a new orthomosaic within the combobox.
        """
        # Stores the orthomosaics in global variable
        self.metashapeOrthomosaics = self.metashapeChunk.orthomosaics

        if len(self.metashapeOrthomosaics) == 0:
            raise Exception("No Orthomosaics found in Chunk")

        # Remove items from existing comboChunk
        self.comboOrthomosaic.clear()

        # Add each orthomosaic name to the combobox
        for orthomosaic in self.metashapeOrthomosaics:
            self.comboOrthomosaic.addItem(str(orthomosaic))

        # Set the orthomosaic to first as default, allow updates
        self.metashapeOrthomosaic = self.metashapeOrthomosaics[0]
        self.comboOrthomosaic.currentIndexChanged.connect(self.metashapeOrthomosaicChanged)

    @pyqtSlot(int)
    def metashapeOrthomosaicChanged(self, index):
        """
        This only runs when a project is loaded (button pressed), or when the user
        selects a new orthomosaic within the combobox.
        """

        # Updates the combobox selection
        if len(self.metashapeOrthomosaics) != 0:
            # Updates the global variable
            self.metashapeOrthomosaic = self.metashapeOrthomosaics[index]

    def loadThumbnails(self):
        """
        This only runs when a project is loaded (button pressed), or when the user
        selects a new chunk within the combobox.
        """

        # A Chunk has been initially loaded or changed
        file_paths = []

        for camera in self.metashapeChunk.cameras:

            try:
                file_path = camera.photo.path
                if os.path.exists(file_path):
                    file_paths.append(file_path)

            except:
                # camera is None
                pass

        if len(file_paths) == 0:
            raise Exception("No valid image paths found in Chunk")

        # Show progress bar
        progress_bar = QtProgressBarCustom()
        progress_bar.setWindowFlags(Qt.ToolTip | Qt.CustomizeWindowHint)
        progress_bar.setWindowModality(Qt.NonModal)
        progress_bar.show()

        progress_bar.setMessage("Creating thumbnails...")

        with ThreadPoolExecutor() as executor:
            futures = []
            for f_idx, file_path in enumerate(file_paths):
                futures.append(executor.submit(create_thumbnail, file_path))
                progress_bar.setProgress(float(f_idx / len(file_paths) * 100.0))
                QApplication.processEvents()

        progress_bar.setMessage("Loading thumbnails...")

        # Reset
        self.thumbnailWidgets = {}

        for f_idx, future in enumerate(futures):
            file_path, thumbnail = future.result()
            thumbnail_widget = ThumbnailWidget(file_path, thumbnail)
            thumbnail_widget.clicked.connect(self.handleThumbnailClick)
            self.thumbnailWidgets[os.path.basename(file_path)] = thumbnail_widget
            progress_bar.setProgress(float(f_idx / len(futures) * 100.0))
            QApplication.processEvents()

        progress_bar.close()
        del progress_bar
        progress_bar = None

    def distance(self, point1, point2):
        """
        Helper function to find distance between two points.
        """
        # Convert points to NumPy arrays for easy computation
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Calculate the Euclidean distance
        dist = np.sqrt(np.sum((point2 - point1) ** 2))

        return dist

    @pyqtSlot()
    def findClosestImage(self, x, y, width, height):
        """
        Metashape API Version for finding the closest images
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        positions = []

        # The current Metashape chunk and orthomosaic
        T = self.metashapeChunk.transform.matrix
        orthomosaic = self.metashapeOrthomosaic

        # For some reason, Metashape Orthomosaic might be off by a few pixels compared
        # to the version that was exported... Using delta for those cases (figure out later)
        delta = 5
        width_difference = np.abs(orthomosaic.width - width)
        height_difference = np.abs(orthomosaic.height - height)

        if width_difference > delta or height_difference > delta:
            QApplication.restoreOverrideCursor()
            msgBox = QMessageBox()
            msgBox.setText(f"Current Map dimensions ({width, height}) do not match "
                           f"selected Orthomosaic ({orthomosaic.width, orthomosaic.height})!")
            msgBox.exec()
            return positions

        # Point p 2D coordinates in ortho CS
        X, Y = (orthomosaic.left + orthomosaic.resolution * x,
                orthomosaic.top - orthomosaic.resolution * y)

        if self.metashapeChunk.elevation:
            # Using the DEM as a surface
            # TODO Use the DEM (and Ortho) from viewerplus instead?
            dem = self.metashapeChunk.elevation
            # Altitude in dem.crs (supposing dem.crs  = ortho.crs)
            Z = dem.altitude(Metashape.Vector((X, Y)))
            # X, Y, Z  point p 3D coordinates  in ortho CS
            if orthomosaic.crs.name[0:17] == 'Local Coordinates':
                # point p in internal coordinate system for case of Local CS
                p = T.inv().mulp(orthomosaic.projection.matrix.inv().mulp(Metashape.Vector((X, Y, Z))))
            else:
                # point p in internal coordinate system (no obstruction test without depth maps)
                p = T.inv().mulp(orthomosaic.crs.unproject(Metashape.Vector((X, Y, Z))))
        else:
            QApplication.restoreOverrideCursor()
            msgBox = QMessageBox()
            msgBox.setText("No DEM in Project")
            msgBox.exec()
            return positions

        for camera in self.metashapeChunk.cameras:

            try:
                # If the point doesn't project, skip
                if not camera.project(p):
                    continue

                u = camera.project(p).x  # u pixel coordinates in camera
                v = camera.project(p).y  # v pixel coordinates in camera

                # Failed the first test, in that the point in not in the image at all
                if u < 0 or u > camera.sensor.width or v < 0 or v > camera.sensor.height:
                    continue

                dist = self.distance(camera.center, p)
                positions.append([camera, u, v, dist])

            except:
                pass

        # Contains **all** cameras that have the view
        positions = np.array(positions)

        if len(positions):

            # Number of thumbnails shown
            N = 15

            # Sort and subset the cameras so that those that are closest are first
            closest_images = positions[np.argsort(positions[:, 3])][0:N]

            # Then sort the closest cameras based on the distance to the center of each camera's image
            sorted_indices = []

            # Calculate and update sorted indices based on the distance to the center of each camera's image
            for i in range(len(closest_images)):
                center_u = closest_images[i, 0].sensor.width / 2
                center_v = closest_images[i, 0].sensor.height / 2

                distance_to_center = np.sqrt((closest_images[i, 1] - center_u) ** 2 +
                                             (closest_images[i, 2] - center_v) ** 2)

                # Append the index along with the distance to the list
                sorted_indices.append((i, distance_to_center))

            # Sort the indices based on the distances
            sorted_indices.sort(key=lambda x: x[1])
            sorted_indices = np.array(sorted_indices).T[0].astype(int)

            # Update closest_images based on the sorted indices
            closest_images = closest_images[sorted_indices]

            # Update the self variable to contain what's needed for
            # viewing in iView (path, u, v, rotation)
            self.closestImages = []

            for closest_image in closest_images:
                # Path to the image
                path = closest_image[0].photo.path
                # Coordinates of the point on image
                u_coord = closest_image[1]
                v_coord = closest_image[2]
                # Rotation angle of image
                transform_matrix = closest_image[0].transform
                rotation_angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
                # Convert the angle from radians to degrees
                rotation = np.degrees(rotation_angle)
                # QT Transform rotates counterclockwise with negative values
                rotation = -rotation if rotation > 0 else rotation

                self.closestImages.append([path, u_coord, v_coord, rotation])

            # Convert to array for transposing
            self.closestImages = np.array(self.closestImages)
            # Update the thumbnails
            self.updateThumbnails(self.closestImages.T[0])
            # Update the viewer to the closest image
            self.handleThumbnailClick(self.closestImages[0][0])

        QApplication.restoreOverrideCursor()

    def updateThumbnails(self, closest_images_paths):
        """
        Removes previous thumbnails within container and populates with new thumbnails;
        updates the viewer and thumbnail selected for current closest image.
        """

        # First clear existing thumbnails
        for i in reversed(range(self.thumbnail_container_layout.count())):
            widgetToRemove = self.thumbnail_container_layout.itemAt(i).widget()
            self.thumbnail_container_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

        # Then add new thumbnails
        for file_path in closest_images_paths:
            # Get a thumbnail widget given the file path
            thumbnail_widget = self.thumbnailWidgets[os.path.basename(file_path)]
            self.thumbnail_container_layout.addWidget(thumbnail_widget)

        # Set the container layout
        self.thumbnail_container.setLayout(self.thumbnail_container_layout)

        # Highlight and display the first image
        self.thumbnail_container_layout.itemAt(0).widget().setSelected(True)

    @pyqtSlot(str)
    def handleThumbnailClick(self, selected_image_path):
        """
        Catches the signal emitted by the ThumbNailWidget mouse event function
        """
        # Update the current image in the scene based on the selected thumbnail
        index = np.where(self.closestImages.T[0] == selected_image_path)[0][0]
        path, u, v, rotation = self.closestImages[index].tolist()
        self.setiViewPreview(path, u, v, rotation)

    def setiViewPreview(self, file_path, u=None, v=None, rotation=None):
        """
        Takes in the path of the image, converts to qimage, displays
        """

        # Reset the Zoom and Rotation
        self.zoom_factor = 1.0
        self.rotation_angle = 0.0

        if not os.path.exists(file_path):
            QApplication.restoreOverrideCursor()
            msgBox = QMessageBox()
            msgBox.setText(f"Image path not found: {file_path}")
            msgBox.exec()
            return

        # Reads the image path, converts to RGB
        image_reader = QImageReader(file_path)
        image = image_reader.read()

        if not None in [u, v]:
            # Add a red dot on the image if coordinates are provided
            self.addRedDot(image, float(u), float(v))

        if rotation:
            # Rotate the image if rotation is provided
            transform = QTransform().rotate(float(rotation))
            image = image.transformed(transform)

        # Display the image using a QPixmap
        new_pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(new_pixmap.scaled(QSize(self.iViewWidth, self.iViewHeight), Qt.KeepAspectRatio))
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def addRedDot(self, image, u, v):
        """
        Fill a region around the specified point (u, v) in the image with red color.
        """
        width, height = image.width(), image.height()

        # Calculate N as a percentage of the image resolution
        N = min(width, height) * 0.50 / 100

        # Calculate the region boundaries
        x_start = max(0, int(u - N / 2))
        y_start = max(0, int(v - N / 2))
        x_end = min(width, int(u + N / 2))
        y_end = min(height, int(v + N / 2))

        # Iterate through the specified region and set pixels to red
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                image.setPixelColor(x, y, QColor(Qt.red))

    def zoom_wheel_event(self, event: QGraphicsSceneWheelEvent):
        """

        """
        # Zoom in or out based on the direction of the wheel
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor /= 1.1  # Zoom out

        # Set the new zoom level
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)

        # Apply the rotation to the transform
        transform.rotate(self.rotation_angle)

        self.graphicsView.setTransform(transform)

    def mouse_press_event(self, event):
        """

        """
        if event.button() == Qt.LeftButton:
            # Start panning if left mouse button is pressed
            self.pan_active = True
            self.pan_start = event.pos()
        elif event.button() == Qt.RightButton:
            # Start rotation if right mouse button is pressed
            self.rotation_active = True
            self.rotation_start = event.pos()

    def mouse_move_event(self, event):
        """

        """
        if self.pan_active:
            # If panning is active, calculate the difference in mouse position and pan accordingly
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()

            # Pan the scene by adjusting the view's scroll bars
            hor_scroll = self.graphicsView.horizontalScrollBar().value() - delta.x()
            ver_scroll = self.graphicsView.verticalScrollBar().value() - delta.y()
            self.graphicsView.horizontalScrollBar().setValue(hor_scroll)
            self.graphicsView.verticalScrollBar().setValue(ver_scroll)

        elif self.rotation_active:
            # If rotation is active, calculate the angle based on the horizontal movement of the mouse
            delta = event.pos() - self.rotation_start
            self.rotation_angle += delta.x()

            # Set the new rotation angle
            transform = QTransform()
            transform.translate(self.original_center.x(), self.original_center.y())
            transform.rotate(self.rotation_angle)
            transform.translate(-self.original_center.x(), -self.original_center.y())

            # Apply the rotation to the transform
            self.graphicsView.setTransform(transform)

            self.rotation_start = event.pos()

    def mouse_release_event(self, event):
        """

        """
        if event.button() == Qt.LeftButton:
            # Stop panning on left mouse button release
            self.pan_active = False
        elif event.button() == Qt.RightButton:
            # Stop rotation on right mouse button release
            self.rotation_active = False

    def resetiView(self):
        """
        Reset the view and the thumbnails.
        """

        self.zoom_factor = 1.0
        self.rotation_angle = 0.0

        # Clear the iView Viewer
        blank_image = QImage(self.iViewWidth, self.iViewHeight, QImage.Format_RGB32)
        blank_image.fill(Qt.black)
        pixmap = QPixmap.fromImage(blank_image)
        # Add the image to the scene
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.graphicsView.setScene(self.scene)

        # Clear the thumbnails
        for i in reversed(range(self.thumbnail_container_layout.count())):
            widgetToRemove = self.thumbnail_container_layout.itemAt(i).widget()
            self.thumbnail_container_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

    def closeEvent(self, event):
        self.closed.emit()
