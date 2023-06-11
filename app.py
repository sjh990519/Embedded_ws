import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Set the window title and size
        self.setWindowTitle("Rock-Paper-Scissors Game")
        self.setGeometry(200, 200, 400, 300)

        # Create the title label
        self.title_label = QLabel("Rock-Paper-Scissors", self)
        self.title_label.setFont(QFont("Arial", 20))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setGeometry(50, 20, 300, 50)

        # Create the buttons
        self.button1 = QPushButton("Single Player", self)
        self.button1.clicked.connect(self.run_file1)
        self.button1.setGeometry(100, 100, 200, 50)

        self.button2 = QPushButton("Dual Player", self)
        self.button2.clicked.connect(self.run_file2)
        self.button2.setGeometry(100, 170, 200, 50)

        self.button3 = QPushButton("Multi Player", self)
        self.button3.clicked.connect(self.run_file3)
        self.button3.setGeometry(100, 240, 200, 50)


    def run_file1(self):
        # Launch the single player game script
        subprocess.Popen(["python", "/home/pi/project_ws/single.py"])

    def run_file2(self):
        # Launch the dual player game script
        subprocess.Popen(["python", "/home/pi/project_ws/dual.py"])

    def run_file3(self):
        # Launch the multi-player game script
        subprocess.Popen(["python", "/home/pi/project_ws/multi.py"])

    def closeEvent(self, event):
        # Perform cleanup and termination steps here
        # For example, you can stop any running processes or perform necessary cleanup operations

        # Call the base class closeEvent to ensure the window is closed properly
        super().closeEvent(event)


if __name__ == '__main__':
    # Create the application and main window
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
