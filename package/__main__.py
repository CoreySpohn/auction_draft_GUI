import sys
from PyQt5.QtWidgets import QApplication
from .mainwindow import MainWindow


def main():
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
