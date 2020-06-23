import sys
sys.path.append("D:\\VS\\VSCode\\Font Recognition")

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import importlib
from Application.model import networkModel

# importlib.import_module('model')

class Ui_MainWindow(object):

    MODEL : networkModel

    def setupUi(self, MainWindow):

        self.MODEL = networkModel()
        self.MODEL.load_model()
        self.MODEL.load_dataset()
        self.MODEL.model.summary()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 500)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(1000, 1000))
        MainWindow.setWindowTitle("Font Identifyer")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(
            self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalFrame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.horizontalFrame.sizePolicy().hasHeightForWidth())
        self.horizontalFrame.setSizePolicy(sizePolicy)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.labelImage = QtWidgets.QLabel(self.horizontalFrame)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.labelImage.sizePolicy().hasHeightForWidth())
        self.labelImage.setSizePolicy(sizePolicy)
        self.labelImage.setMinimumSize(QtCore.QSize(0, 0))
        self.labelImage.setMaximumSize(QtCore.QSize(500, 500))
        self.labelImage.setText("")
        self.labelImage.setScaledContents(True)
        self.labelImage.setObjectName("labelImage")
        self.verticalLayout_2.addWidget(
            self.labelImage, 0, QtCore.Qt.AlignHCenter)
        self.buttonFile = QtWidgets.QPushButton(self.horizontalFrame)
        self.buttonFile.setObjectName("buttonFile")
        self.buttonFile.clicked.connect(self.buttonFileHandler)
        self.verticalLayout_2.addWidget(self.buttonFile)
        self.buttonProcess = QtWidgets.QPushButton(self.horizontalFrame)
        self.buttonProcess.setObjectName("buttonProcess")
        self.buttonProcess.clicked.connect(self.buttonProcessHandler)
        self.buttonProcess.setEnabled(False)
        self.verticalLayout_2.addWidget(self.buttonProcess)
        spacerItem = QtWidgets.QSpacerItem(
            150, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem)
        self.boxResults = QtWidgets.QTextBrowser(self.horizontalFrame)
        self.boxResults.setMinimumSize(QtCore.QSize(400, 0))
        self.boxResults.setAutoFormatting(QtWidgets.QTextEdit.AutoBulletList)
        self.boxResults.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard | QtCore.Qt.LinksAccessibleByMouse |
                                                QtCore.Qt.TextBrowserInteraction | QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse)
        self.boxResults.setObjectName("boxResults")
        self.verticalLayout_2.addWidget(self.boxResults)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout.addWidget(
            self.horizontalFrame, 0, QtCore.Qt.AlignHCenter)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.buttonFile.setText(_translate("MainWindow", "Browse Images"))
        self.buttonProcess.setText(_translate("MainWindow", "Identify Font"))
        self.boxResults.setText(_translate("MainWindow", ""))

    def buttonFileHandler(self):
        fileDialog = QFileDialog.getOpenFileName(
            self.labelImage, 'Open File', '', 'Images (*.png *.gif *.jpg)')
        self.path = fileDialog[0]
        pixmap = QtGui.QPixmap(self.path)
        if(pixmap.isNull() == False):
            self.labelImage.setPixmap(pixmap)
            self.buttonProcess.setEnabled(True)
        else:
            self.labelImage.setPixmap(pixmap)
            self.buttonProcess.setEnabled(False)

    def buttonProcessHandler(self):
        #edit here..........................................................................................
        _translate = QtCore.QCoreApplication.translate
        result = self.MODEL.predict(self.path)
        self.boxResults.setText(_translate("MainWindow", result))
        #self.boxResults.setSource(QtCore.QUrl("file:///C:/path/to/result.txt"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
