# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(852, 448)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.selectedPlayerLabel = QtWidgets.QLabel(Form)
        self.selectedPlayerLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.selectedPlayerLabel.setObjectName("selectedPlayerLabel")
        self.gridLayout.addWidget(self.selectedPlayerLabel, 0, 0, 1, 1)
        self.draftBoardLabel = QtWidgets.QLabel(Form)
        self.draftBoardLabel.setScaledContents(True)
        self.draftBoardLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.draftBoardLabel.setObjectName("draftBoardLabel")
        self.gridLayout.addWidget(self.draftBoardLabel, 0, 1, 1, 1)
        self.optTeamLabel = QtWidgets.QLabel(Form)
        self.optTeamLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.optTeamLabel.setObjectName("optTeamLabel")
        self.gridLayout.addWidget(self.optTeamLabel, 2, 2, 1, 1)
        self.myTeamTable = QtWidgets.QTableWidget(Form)
        self.myTeamTable.setObjectName("myTeamTable")
        self.myTeamTable.setColumnCount(0)
        self.myTeamTable.setRowCount(0)
        self.gridLayout.addWidget(self.myTeamTable, 1, 2, 1, 1)
        self.myTeamLabel = QtWidgets.QLabel(Form)
        self.myTeamLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.myTeamLabel.setObjectName("myTeamLabel")
        self.gridLayout.addWidget(self.myTeamLabel, 0, 2, 1, 1)
        self.optTeamTable = QtWidgets.QTableWidget(Form)
        self.optTeamTable.setObjectName("optTeamTable")
        self.optTeamTable.setColumnCount(0)
        self.optTeamTable.setRowCount(0)
        self.gridLayout.addWidget(self.optTeamTable, 3, 2, 1, 1)
        self.draftBoard = QtWidgets.QTableWidget(Form)
        self.draftBoard.setObjectName("draftBoard")
        self.draftBoard.setColumnCount(0)
        self.draftBoard.setRowCount(0)
        self.gridLayout.addWidget(self.draftBoard, 1, 1, 4, 1)
        self.selectedPlayerTable = QtWidgets.QTableWidget(Form)
        self.selectedPlayerTable.setObjectName("selectedPlayerTable")
        self.selectedPlayerTable.setColumnCount(0)
        self.selectedPlayerTable.setRowCount(0)
        self.gridLayout.addWidget(self.selectedPlayerTable, 2, 0, 3, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.selectedPlayerLabel.setText(_translate("Form", "Currently Up for Auction"))
        self.draftBoardLabel.setText(_translate("Form", "Draft Board"))
        self.optTeamLabel.setText(_translate("Form", "Optimal Team"))
        self.myTeamLabel.setText(_translate("Form", "My Team"))