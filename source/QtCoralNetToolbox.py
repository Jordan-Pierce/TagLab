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

from source.tools.CoralNetToolbox.API import api
from source.tools.CoralNetToolbox.Upload import upload


class CoralNetToolboxWidget(QWidget):

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super(CoralNetToolboxWidget, self).__init__(parent)

        # Parameters
        self.username = ""
        self.password = ""
        self.source_id_1 = ""
        self.source_id_2 = ""
        self.output_folder = ""

        # --------------------
        # The window settings
        # --------------------
        self.setWindowTitle("CoralNet Toolbox")

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

        # -----------------------
        # CoralNet Parameters
        # -----------------------

        # Username
        layoutUsername = QHBoxLayout()
        layoutUsername.setAlignment(Qt.AlignLeft)
        self.lblUsername = QLabel("Username: ")
        self.lblUsername.setFixedWidth(130)
        self.lblUsername.setMinimumWidth(130)
        self.editUsername = QLineEdit("")
        self.editUsername.setPlaceholderText("CoralNet Username")
        self.editUsername.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnUsername = QPushButton("...")
        self.btnUsername.setFixedWidth(20)
        self.btnUsername.setMinimumWidth(20)
        self.btnUsername.clicked.connect(self.getCoralNetUsername)

        layoutUsername.addWidget(self.lblUsername)
        layoutUsername.addWidget(self.editUsername)
        layoutUsername.addWidget(self.btnUsername)

        # Password
        layoutPassword = QHBoxLayout()
        layoutPassword.setAlignment(Qt.AlignLeft)
        self.lblPassword = QLabel("Password: ")
        self.lblPassword.setFixedWidth(130)
        self.lblPassword.setMinimumWidth(130)
        self.editPassword = QLineEdit("")
        self.editPassword.setEchoMode(QLineEdit.Password)
        self.editPassword.setPlaceholderText("CoralNet Password")
        self.editPassword.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnPassword = QPushButton("...")
        self.btnPassword.setFixedWidth(20)
        self.btnPassword.setMinimumWidth(20)
        self.btnPassword.clicked.connect(self.getCoralNetPassword)

        layoutPassword.addWidget(self.lblPassword)
        layoutPassword.addWidget(self.editPassword)
        layoutPassword.addWidget(self.btnPassword)

        # Source ID 1 (images)
        layoutSourceID1 = QHBoxLayout()
        layoutSourceID1.setAlignment(Qt.AlignLeft)
        self.lblSourceID1 = QLabel("Source ID 1: ")
        self.lblSourceID1.setFixedWidth(130)
        self.lblSourceID1.setMinimumWidth(130)
        self.editSourceID1 = QLineEdit("")
        self.editSourceID1.setPlaceholderText("Source ID for storing images")
        self.editSourceID1.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutSourceID1.addWidget(self.lblSourceID1)
        layoutSourceID1.addWidget(self.editSourceID1)

        # Source ID 2 (model)
        layoutSourceID2 = QHBoxLayout()
        layoutSourceID2.setAlignment(Qt.AlignLeft)
        self.lblSourceID2 = QLabel("Source ID 2: ")
        self.lblSourceID2.setFixedWidth(130)
        self.lblSourceID2.setMinimumWidth(130)
        self.editSourceID2 = QLineEdit("")
        self.editSourceID2.setPlaceholderText("Source ID for desired model")
        self.editSourceID2.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutSourceID2.addWidget(self.lblSourceID2)
        layoutSourceID2.addWidget(self.editSourceID2)

        # Output Folder
        layoutOutputFolder = QHBoxLayout()
        layoutOutputFolder.setAlignment(Qt.AlignLeft)
        self.lblOutputFolder = QLabel("Output Folder: ")
        self.lblOutputFolder.setFixedWidth(130)
        self.lblOutputFolder.setMinimumWidth(130)
        self.editOutputFolder = QLineEdit("")
        self.editOutputFolder.setPlaceholderText("")
        self.editOutputFolder.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnOutputFolder = QPushButton("...")
        self.btnOutputFolder.setFixedWidth(20)
        self.btnOutputFolder.setMinimumWidth(20)
        self.btnOutputFolder.clicked.connect(self.chooseOutputFolder)

        layoutOutputFolder.addWidget(self.lblOutputFolder)
        layoutOutputFolder.addWidget(self.editOutputFolder)
        layoutOutputFolder.addWidget(self.btnOutputFolder)

        # -----------------------
        # Buttons
        # -----------------------
        layoutButtons = QHBoxLayout()
        layoutButtons.setAlignment(Qt.AlignRight)
        layoutButtons.addStretch()

        self.btnApply = QPushButton("Apply")
        self.btnApply.clicked.connect(self.apply)
        self.btnExit = QPushButton("Exit")
        self.btnExit.clicked.connect(self.close)

        layoutButtons.addWidget(self.btnApply)
        layoutButtons.addWidget(self.btnExit)

        # -----------------------
        # Final Layout order
        # -----------------------
        layoutV = QVBoxLayout()

        layoutV.addLayout(layoutUsername)
        layoutV.addLayout(layoutPassword)
        layoutV.addLayout(layoutSourceID1)
        layoutV.addLayout(layoutSourceID2)
        layoutV.addLayout(layoutOutputFolder)
        layoutV.setSpacing(3)
        layoutV.addLayout(layoutButtons)

        self.setLayout(layoutV)

    @pyqtSlot()
    def getCoralNetUsername(self):
        """

        """
        coralnetUsername = os.getenv('CORALNET_USERNAME')
        if coralnetUsername:
            self.editUsername.setText(coralnetUsername)

    @pyqtSlot()
    def getCoralNetPassword(self):
        """

        """
        coralnetPassword = os.getenv('CORALNET_PASSWORD')
        if coralnetPassword:
            self.editPassword.setText(coralnetPassword)

    @pyqtSlot()
    def chooseOutputFolder(self):
        """

        """
        folder_name = QFileDialog.getExistingDirectory(self, "Choose a Folder as the root working directory", "")
        if folder_name:
            self.editOutputFolder.setText(folder_name)

    @pyqtSlot()
    def apply(self):
        """

        """
        self.username = self.editUsername.text()
        self.password = self.editPassword.text()
        self.source_id_1 = self.editSourceID1.text()
        self.source_id_2 = self.editSourceID2.text()
        self.output_folder = self.editOutputFolder.text()

        # Cache the Username and Password as local variables (after authentication)?
        os.environ["CORALNET_USERNAME"] = self.username
        os.environ["CORALNET_PASSWORD"] = self.password

        print("Done.")
        self.close()

    def close(self):
        self.closed.emit()