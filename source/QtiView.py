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

from PyQt5.QtCore import Qt, QSize, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtWidgets import QSizePolicy, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QTabWidget
from PyQt5.QtWidgets import QGroupBox, QWidget, QFileDialog, QComboBox, QApplication, QMessageBox

import cv2
import numpy as np

import Metashape

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version,
                                                                      compatible_major_version))


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
        self.closestImage = None
        self.closestImages = []

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

        # Create a tab widget
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
        self.lblLicense.setFixedWidth(125)
        self.lblLicense.setMinimumWidth(125)
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
        self.lblProject.setFixedWidth(125)
        self.lblProject.setMinimumWidth(125)
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
        self.lblChunk.setFixedWidth(125)
        self.lblChunk.setMinimumWidth(125)
        self.comboChunk = QComboBox()
        self.comboChunk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.comboChunk.setMinimumWidth(200)  # Adjust the minimum width as needed
        layoutChunk.addWidget(self.lblChunk)
        layoutChunk.addWidget(self.comboChunk)

        # Metashape Orthomosaic (combobox)
        layoutOrthomosaic = QHBoxLayout()
        layoutOrthomosaic.setAlignment(Qt.AlignLeft)
        self.lblOrthomosaic = QLabel("Orthomosaic: ")
        self.lblOrthomosaic.setFixedWidth(125)
        self.lblOrthomosaic.setMinimumWidth(125)
        self.comboOrthomosaic = QComboBox()
        self.comboOrthomosaic.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.comboOrthomosaic.setMinimumWidth(200)  # Adjust the minimum width as needed
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

        # ---------------------
        # iView panel
        # ---------------------
        layoutiView = QVBoxLayout()

        # Combobox for closest images
        layoutClosestCombo = QHBoxLayout()

        self.lblClosestImages = QLabel("Current Image: ")
        self.comboClosestImages = QComboBox()
        self.comboClosestImages.setMinimumWidth(300)

        for closestImage in self.closestImages:
            self.comboClosestImages.addItem(closestImage)

        self.comboClosestImages.currentIndexChanged.connect(self.closestImageChanged)

        layoutClosestCombo.setAlignment(Qt.AlignRight)
        layoutClosestCombo.addWidget(self.lblClosestImages)
        layoutClosestCombo.addWidget(self.comboClosestImages)

        # Preview image of current image
        self.iViewWidth = 1200
        self.iViewHeight = 675
        self.QlabelRGB = QLabel("")
        self.QPixmapRGB = QPixmap(self.iViewWidth, self.iViewHeight)
        self.QPixmapRGB.fill(Qt.black)
        self.QlabelRGB.setPixmap(self.QPixmapRGB)

        # Set maximum size for the QLabel containing the image
        self.QlabelRGB.setMaximumSize(self.iViewWidth, self.iViewHeight)
        self.QlabelRGB.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layoutClosestPreview = QHBoxLayout()
        layoutClosestPreview.setAlignment(Qt.AlignCenter)
        layoutClosestPreview.addWidget(self.QlabelRGB)

        # Add the combobox and preview to iView layout
        layoutiView.addLayout(layoutClosestCombo)
        layoutiView.addLayout(layoutClosestPreview)

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

        # Metashape and VISCORE Panels
        layoutV.addWidget(self.tabWidget)
        layoutV.addSpacing(10)

        # iView Panel (widget)
        layoutV.addWidget(self.iViewPanel)

        # Close Panel (button)
        layoutV.addLayout(layoutClose)
        layoutV.setSpacing(3)

        self.setLayout(layoutV)

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

        # Good or Bad box
        msgBox = QMessageBox()

        try:
            # Get the metashape license, try to activate
            metashape_license = self.editLicense.text()

            assert metashape_license != ""

            # Set the metashape license
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
            # Project has been opened, get the chunks
            self.metashapeChunks = self.metashapeProject.chunks

            if len(self.metashapeChunks) == 0:
                QApplication.restoreOverrideCursor()
                msgBox.setText("No Chunks found in Project")
                msgBox.exec()
                return

            # Remove items from comboChunk (in case of project change)
            self.comboChunk.clear()

            # Add each chunk name to the combobox
            for chunk in self.metashapeChunks:
                self.comboChunk.addItem(chunk.label)

            # Set the chunk to first as default, allow updates
            self.metashapeChunk = self.metashapeChunks[0]
            self.comboChunk.currentIndexChanged.connect(self.metashapeChunksChanged)

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText("Failed to load Chunks!")
            msgBox.exec()
            return

        try:
            # Chunk has been opened, get the orthomosaics
            self.metashapeOrthomosaics = self.metashapeChunk.orthomosaics

            if len(self.metashapeOrthomosaics) == 0:
                QApplication.restoreOverrideCursor()
                msgBox.setText("No Orthomosaics found in Chunk")
                msgBox.exec()
                return

            # Remove items from comboOrthomosaic (in case of project change)
            self.comboOrthomosaic.clear()

            # Add each orthomosaic name to the combobox
            for orthomosaic in self.metashapeOrthomosaics:
                self.comboOrthomosaic.addItem(str(orthomosaic))

            # Set the orthomosaic to first as default, allow updates
            self.metashapeOrthomosaic = self.metashapeOrthomosaics[0]
            self.comboOrthomosaic.currentIndexChanged.connect(self.metashapeOrthomosaicsChanged)

        except Exception as e:
            # Show a bad box
            QApplication.restoreOverrideCursor()
            msgBox.setText("Failed to load Chunks!")
            msgBox.exec()
            return

        QApplication.restoreOverrideCursor()

    @pyqtSlot(int)
    def metashapeChunksChanged(self, index):
        """

        """
        if len(self.metashapeChunks) != 0:
            self.metashapeChunk = self.metashapeChunks[index]

    @pyqtSlot(int)
    def metashapeOrthomosaicsChanged(self, index):
        """

        """
        if len(self.metashapeOrthomosaics) != 0:
            self.metashapeOrthomosaic = self.metashapeOrthomosaics[index]

    def distance(self, point1, point2):
        """

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

        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        positions = []

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

            N = 100
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

            # Update the image combobox
            self.closestImagesChanged(closest_images)

        QApplication.restoreOverrideCursor()

    def closestImagesChanged(self, closest_images):
        """
        Update the images combobox
        """

        # Set the new closest image(s)
        self.closestImages = closest_images

        # Remove items from comboClosestImages
        self.comboClosestImages.clear()

        # Add each image name to the combobox
        for image in self.closestImages:
            self.comboClosestImages.addItem(str(image[0].label))

    @pyqtSlot(int)
    def closestImageChanged(self, index):
        """

        """
        self.comboClosestImages.setFocus()

        if 0 <= index < len(self.closestImages):
            self.closestImage = self.closestImages[index]
            self.setiViewPreview(self.closestImage)

    def setiViewPreview(self, item):
        """
        Takes in the path of the image, opens with cv2, converts to qimage, displays
        """

        path = item[0].photo.path

        if not os.path.exists(path):
            QApplication.restoreOverrideCursor()
            msgBox = QMessageBox()
            msgBox.setText(f"Image path not found: {path}")
            msgBox.exec()
            return

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pixel that corresponds to user's mouse, in image
        x, y = int(item[1]), int(item[2])

        # Define the size of the area to be modified
        area_size = 15

        # Set the specified region to red
        image[y - area_size: y + area_size, x - area_size: x + area_size, :] = [255, 0, 0]

        height, width, channel = image.shape
        bytes_per_line = 3 * width  # Assuming 3 channels (RGB)

        # Create QImage from the NumPy array data
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.QPixmapRGB = QPixmap.fromImage(qimage)
        self.QlabelRGB.setPixmap(self.QPixmapRGB.scaled(QSize(self.iViewWidth, self.iViewHeight), Qt.KeepAspectRatio))

    def keyPressEvent(self, event):
        """

        """
        self.comboClosestImages.setFocus()

        if event.key() == Qt.Key_Left:
            # Handle left arrow key
            if self.comboClosestImages.currentIndex() > 0:
                self.comboClosestImages.setCurrentIndex(self.comboClosestImages.currentIndex() - 1)
        elif event.key() == Qt.Key_Right:
            # Handle right arrow key
            if self.comboClosestImages.currentIndex() < self.comboClosestImages.count() - 1:
                self.comboClosestImages.setCurrentIndex(self.comboClosestImages.currentIndex() + 1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
