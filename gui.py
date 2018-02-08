# GUI for the style transfer

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot

 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'theprintspace fast style transfer'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):

        # window settings
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # self.setFixedSize(600, 600)

        # content and style
        self.content = QLabel(self)
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.jpg *.gif)")
        pixmap = QPixmap(fname[0])
        pixmap.scaledToWidth(320)
        self.content.setPixmap(pixmap)
        self.content.resize(pixmap.width(),pixmap.height())

 
        # Create a button in the window
        self.button = QPushButton('START', self)
        self.button.move(20,160)
        self.button.resize(280, 20)
 
        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()
 
    @pyqtSlot()
    def on_click(self):
        print("Yeah!")
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())