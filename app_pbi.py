import sys
import subprocess
import webbrowser
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setWindowTitle("Foot fall System")
        self.resize(400, 150)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        description_label = QLabel("Welcome to the footfall analysis system", self)
        description_label.setAlignment(Qt.AlignCenter)

        # Set the font size of the description label
        font = QFont()
        font.setPointSize(16)  # Change 16 to the desired font size
        description_label.setFont(font)

        # Set the font color of the description label
        description_label.setStyleSheet("color: blue;")  # Change "blue" to your desired color

        self.layout.addWidget(description_label)

        button_layout1 = QHBoxLayout()
        self.start_button1 = QPushButton("Start A_01", self)
        self.stop_button1 = QPushButton("Stop A_01", self)
        button_layout1.addWidget(self.start_button1)
        button_layout1.addWidget(self.stop_button1)

        button_layout2 = QHBoxLayout()
        self.start_button2 = QPushButton("Start  A_02", self)
        self.stop_button2 = QPushButton("Stop A_02", self)
        button_layout2.addWidget(self.start_button2)
        button_layout2.addWidget(self.stop_button2)

        button_layout3 = QHBoxLayout()
        self.power_bi_button = QPushButton("Open Dashboard", self)
        button_layout3.addWidget(self.power_bi_button)

        self.layout.addLayout(button_layout1)
        self.layout.addLayout(button_layout2)
        self.layout.addLayout(button_layout3)

        self.start_button1.clicked.connect(self.start_script1)
        self.stop_button1.clicked.connect(self.stop_script1)
        self.start_button2.clicked.connect(self.start_script2)
        self.stop_button2.clicked.connect(self.stop_script2)
        self.power_bi_button.clicked.connect(self.open_power_bi_dashboard)

        self.process1 = None
        self.process2 = None

    def start_script1(self):
        if self.process1 is None or self.process1.poll() is not None:
            self.process1 = subprocess.Popen(["python", "counting_time_diff_sec_db.py"])

    def stop_script1(self):
        if self.process1 and self.process1.poll() is None:
            self.process1.terminate()

    def start_script2(self):
        if self.process2 is None or self.process2.poll() is not None:
            self.process2 = subprocess.Popen(["python", "counting_dbl_side_diff_sec_db.py"])

    def stop_script2(self):
        if self.process2 and self.process2.poll() is not None:
            self.process2.terminate()

    def open_power_bi_dashboard(self):
        dashboard_url = 'https://app.powerbi.com/groups/me/reports/739d16d1-0aa6-44da-b2e8-71c014acc472/ReportSection15dbfb33205b066302e7?experience=power-bi'
        webbrowser.open(dashboard_url)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MyWindow()
    mainWin.show()
    sys.exit(app.exec_())
