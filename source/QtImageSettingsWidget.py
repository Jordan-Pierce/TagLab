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
from datetime import datetime

from PyQt5.QtCore import Qt, QSize, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QImage, QImageReader, QPixmap, QIcon, qRgb, qRed, qGreen, qBlue
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QComboBox, QSizePolicy
from PyQt5.QtWidgets import QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from source import utils


class QtImageSettingsWidget(QWidget):
    accepted = pyqtSignal()

    def __init__(self, parent=None):
        super(QtImageSettingsWidget, self).__init__(parent)

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setMinimumWidth(300)
        self.setMinimumHeight(100)

        TEXT_SPACE = 100

        self.fields = {
            "rgb_filenames": {"name": "Image(s):", "value": "", "place": "Path of image(s)", "width": 300,
                             "action": self.chooseImageFiles},
            "acquisition_dates": {"name": "Acquisition Date:", "value": "", "place": "YYYY-MM-DD", "width": 150,
                                 "action": None},
        }
        self.data = {}

        layoutV = QVBoxLayout()

        for key, field in self.fields.items():
            label = QLabel(field["name"])
            label.setFixedWidth(TEXT_SPACE)
            label.setAlignment(Qt.AlignRight)
            label.setMinimumWidth(150)

            edit = QLineEdit(field["value"])
            edit.setStyleSheet("background-color: rgb(55,55,55); border: 1px solid rgb(90,90,90)")
            edit.setMinimumWidth(field["width"])
            edit.setPlaceholderText(field["place"])
            edit.setMaximumWidth(20)
            field["edit"] = edit

            button = None
            if field["action"] is not None:
                button = QPushButton("...")
                button.setMaximumWidth(20)
                button.clicked.connect(field["action"])
                field["button"] = button

            layout = QHBoxLayout()
            layout.setAlignment(Qt.AlignLeft)
            layout.addWidget(label)
            layout.addWidget(edit)
            if button is not None:
                layout.addWidget(button)
            layout.addStretch()
            layoutV.addLayout(layout)

        buttons_layout = QHBoxLayout()

        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.close)
        self.btnApply = QPushButton("Apply")
        self.btnApply.clicked.connect(self.accept)

        buttons_layout.setAlignment(Qt.AlignRight)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btnCancel)
        buttons_layout.addWidget(self.btnApply)

        ###########################################################

        layoutV.addLayout(buttons_layout)
        self.setLayout(layoutV)

        self.setWindowTitle("Image Settings")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)

    def getNow(self):
        """

        """
        # Get the current date and time
        current_time = datetime.now()
        return current_time.strftime("%Y-%m-%d")

    def disableRGBloading(self):
        self.fields["rgb_filenames"]["edit"].setEnabled(False)
        self.fields["rgb_filenames"]["button"].hide()

    def enableRGBloading(self):
        self.fields["rgb_filenames"]["edit"].setEnabled(True)
        self.fields["rgb_filenames"]["button"].show()

    @pyqtSlot()
    def chooseImageFiles(self):
        filters = "Image (*.png *.jpg *.jpeg *.tif *.tiff)"
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Input Image Files", "", filters)
        if fileNames:
            joinedFileNames = " ".join(fileNames)
            self.fields["rgb_filenames"]["edit"].setText(joinedFileNames)

    @pyqtSlot()
    def accept(self):

        for key, field in self.fields.items():
            self.data[key] = field["edit"].text()

        # Check validity of the acquisition date
        acquisition_date = self.data["acquisition_dates"]

        # If none was provided, create one for now
        if acquisition_date == "":
            acquisition_date = self.getNow()

        if not utils.isValidDate(acquisition_date):
            msgBox = QMessageBox()
            msgBox.setText("Invalid date format. Please, enter the acquisition date as YYYY-MM-DD.")
            msgBox.exec()
            return

        # Parse each of the filenames
        self.data['rgb_filenames'] = self.data['rgb_filenames'].split(" ")
        rgb_filenames = self.data['rgb_filenames']

        # Update data fields
        self.data['names'] = []
        self.data['acquisition_dates'] = []
        self.data['count'] = 0

        # Loop through each
        for rgb_filename in rgb_filenames:

            # Check if each of the RGB image file exists
            if not os.path.exists(rgb_filename):
                msgBox = QMessageBox()
                msgBox.setText("The RGB image file does not seems to exist.")
                msgBox.exec()
                return

            else:
                # Set the name, and acquisition date for each of the files
                self.data['names'].append(".".join(os.path.basename(rgb_filename).split(".")[0:-1]))
                self.data['acquisition_dates'].append(acquisition_date)

                # Update the count
                self.data['count'] += 1

        # TODO: redundant check, remove it ?
        for rgb_filename in rgb_filenames:
            image_reader = QImageReader(rgb_filename)
            size = image_reader.size()
            if size.width() > 32767 or size.height() > 32767:
                msgBox = QMessageBox()
                msgBox.setText(f"The image {os.path.basename(rgb_filename)} is too big. "
                               f"TagLab is limited to 32767x32767 pixels.")
                msgBox.exec()
                return

        self.accepted.emit()
        self.close()