# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:37:50 2021

@author: adamkritz
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication

from PyQt5.QtWidgets import QSizePolicy

from PyQt5.QtWidgets import QCheckBox    # checkbox
from PyQt5.QtWidgets import QPushButton  # pushbutton
from PyQt5.QtWidgets import QLineEdit    # Lineedit
from PyQt5.QtWidgets import QRadioButton # Radio Buttons
from PyQt5.QtWidgets import QGroupBox    # Group Box

from numpy.polynomial.polynomial import polyfit
import numpy as np

#----------------------------------------------------------------------
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

from PyQt5.QtCore import QSize 



# These components are essential for creating the graphics in pqt5 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
from matplotlib.figure import Figure 



class NumericalVars(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(NumericalVars, self).__init__()

        self.Title = 'Numerical Variables'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)
        
        self.groupBox1 = QGroupBox('Variables')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.groupBox2 = QGroupBox('Description')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)# Creates vertical layout

        self.b1 = QRadioButton("Duration")
        self.b1.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b1)

        self.b2 = QRadioButton("Budget")
        self.b2.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b2)

        self.b3 = QRadioButton("World Wide Gross Income")
        self.b3.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b3)
        
        self.b4 = QRadioButton("USA Gross Income")
        self.b4.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b4)
        
        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.b3)
        self.groupBox1Layout.addWidget(self.b4)
        
        
        
        self.label = QLabel("")
        self.layout.addWidget(self.label)
        self.groupBox2Layout.addWidget(self.label)
    

        # figure and canvas figure to draw the graph is created to

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        
        self.groupBox3Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addStretch(1)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(1000, 1000)                         # Resize the window

    def onClicked(self):
        if self.b1.isChecked():
            self.label.setText('The length of the movie')
            self.ax1.clear()
            self.ax1.scatter([1,2,3,4,5,6,7,8,9,10], [15,25,30,20,50,55,60,55,70,75])
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b2.isChecked():
            self.label.setText('The budget for the movie')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b3.isChecked():
            self.label.setText('The amount of money the movie made world-wide')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b4.isChecked():
            self.label.setText('The amount of money the movie made in the United States')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

            
   
class CategoricalVars(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(CategoricalVars, self).__init__()

        self.Title = 'Categorical Variables'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout
        
        self.groupBox1 = QGroupBox('Variables')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.groupBox2 = QGroupBox('Description')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)# Creates vertical layout

        self.b1 = QRadioButton("Title")
        self.b1.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b1)

        self.b2 = QRadioButton("Date Published")
        self.b2.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b2)

        self.b3 = QRadioButton("Genre")
        self.b3.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b3)
        
        self.b5 = QRadioButton("Country")
        self.b5.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b5)
        
        self.b6 = QRadioButton("Director")
        self.b6.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b6)
        
        self.b7 = QRadioButton("Writer")
        self.b7.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b7)
        
        self.b8 = QRadioButton("Production Company")
        self.b8.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b8)
        
        self.b9 = QRadioButton("Actors")
        self.b9.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b9)
        
        self.b10 = QRadioButton("Description")
        self.b10.toggled.connect(self.onClicked2)
        self.layout.addWidget(self.b10)
        
        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.b3)
        self.groupBox1Layout.addWidget(self.b5)
        self.groupBox1Layout.addWidget(self.b6)
        self.groupBox1Layout.addWidget(self.b7)
        self.groupBox1Layout.addWidget(self.b8)
        self.groupBox1Layout.addWidget(self.b9)
        self.groupBox1Layout.addWidget(self.b10)
        
        self.label = QLabel("")
        self.layout.addWidget(self.label)
        self.groupBox2Layout.addWidget(self.label)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        
        self.groupBox3Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addStretch(1)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(1000, 1000)                      # Resize the window


    def onClicked2(self):
        if self.b1.isChecked():
            self.label.setText('The amount of words in the title of the movie')
            self.ax1.clear()
            self.ax1.scatter([1,2,3,4,5,6,7,8,9,10], [15,25,30,20,50,55,60,55,70,75])
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b2.isChecked():
            self.label.setText('The date the movie released')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b3.isChecked():
            self.label.setText('The genre of the movie')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b5.isChecked():
            self.label.setText('The country the movie was initially released in')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b6.isChecked():
            self.label.setText('description for director')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b7.isChecked():
            self.label.setText('description for writer')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b8.isChecked():
            self.label.setText('description for prodcution company')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b9.isChecked():
            self.label.setText('description for actors')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b10.isChecked():
            self.label.setText('description for description')
            self.ax1.clear()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
class TargetVar(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(TargetVar, self).__init__()
        
        self.Title = 'Weighted Average Vote'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        
        self.groupBox1 = QGroupBox('Description')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.groupBox2 = QGroupBox('Graphic')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.label = QLabel("The average vote for an IMDb movie is calculated by the averaging all the ratings for a movie. However, IMDb uses weighted average vote over raw average. \nThis allows IMDb to weight votes differently in order to detect unusual activity, like review-bombing. This allows IMDb to prevent users from drastically changing a movie's score.")
        self.layout.addWidget(self.label)
        self.groupBox1Layout.addWidget(self.label)
        
        
        
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        
        self.groupBox2Layout.addWidget(self.canvas)
        
        self.ax1.clear()
        self.ax1.scatter([1,2,3,4,5,6,7,8,9,10], [15,25,30,20,50,55,60,55,70,75])
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
    
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)                      # Resize the window


class SGD(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(SGD, self).__init__()
        
        self.Title = 'SGD Regressor'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  
        
class RF(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(RF, self).__init__()
        
        self.Title = 'Random Forest Regressor'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  


class Gradient(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(Gradient, self).__init__()
        
        self.Title = 'Gradient Boosting Regressor'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  


class Ada(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(Ada, self).__init__()
        
        self.Title = 'AdaBoost Regressor'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  


class lin(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(lin, self).__init__()
        
        self.Title = 'Linear Regression'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  


class predi(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(predi, self).__init__()
        
        self.Title = 'Predicition Tool'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 500)  


class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        # set size
        self.left = 300
        self.top = 300
        self.width = 700
        self.height = 700

        # Title

        self.Title = 'GUI for project'

        #call intiUI to create elements for menu

        self.initUI()
        
    def initUI(self):

        
        # Creates the menu and the items
       
        
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        
        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('File')
        
        preproc = mainMenu.addMenu('EDA') 
        
        model = mainMenu.addMenu('Modelling') 
        
        pred = mainMenu.addMenu('Prediction Tool') 
        

        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        
        # exit tabs
        
        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)
        
        # preprocessing tabs
        
        preproc1Button = QAction("Numerical Variables", self)   
        preproc1Button.setStatusTip("All the numeric variables we used")   
        preproc1Button.triggered.connect(self.preproc1)    

        preproc2Button = QAction("Categorical Variables", self)   
        preproc2Button.setStatusTip("All the categorical variables we used")  
        preproc2Button.triggered.connect(self.preproc2)
        
        preproc3Button = QAction("Target Variable", self)   
        preproc3Button.setStatusTip("The target variable")  
        preproc3Button.triggered.connect(self.preproc3)

        preproc.addAction(preproc1Button)    
        preproc.addAction(preproc2Button)
        preproc.addAction(preproc3Button)
        
        # modelling
        
        model1button = QAction('SGD Regressor',  self)
        model1button.setStatusTip("SGD Regressor Model")   
        model1button.triggered.connect(self.model1)  
        
        model2button = QAction('Random Forest Regressor',  self)
        model2button.setStatusTip("Random Forest Regressor Model")   
        model2button.triggered.connect(self.model2)  
        
        model3button = QAction('Gradient Boosting Regressor',  self)
        model3button.setStatusTip("Gradient Boosting Regressor Model")   
        model3button.triggered.connect(self.model3)  
        
        model4button = QAction('AdaBoost Regressor',  self)
        model4button.setStatusTip("AdaBoost Regressor Model")   
        model4button.triggered.connect(self.model4)  
        
        model5button = QAction('Linear Regression',  self)
        model5button.setStatusTip("Linear Regression Regressor")   
        model5button.triggered.connect(self.model5)  

        model.addAction(model1button)
        model.addAction(model2button)
        model.addAction(model3button)
        model.addAction(model4button)
        model.addAction(model5button)

        pred1button = QAction("Let's Predict",  self)
        pred1button.setStatusTip("This tool will make a prediction for a movie")   
        pred1button.triggered.connect(self.pred1) 
        
        pred.addAction(pred1button)
        
        # This line shows the windows
        
        self.dialogs = list()

        self.show()
    
        
    def preproc1(self):
        dialog = NumericalVars()
        self.dialogs.append(dialog) 
        dialog.show()  
        
    def preproc2(self):
        dialog = CategoricalVars()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def preproc3(self):
        dialog = TargetVar()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def model1(self):
        dialog = SGD()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model2(self):
        dialog = RF()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model3(self):
        dialog = Gradient()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model4(self):
        dialog = Ada()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def model5(self):
        dialog = lin()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def pred1(self):
        dialog = predi()
        self.dialogs.append(dialog) 
        dialog.show()
      

#::------------------------
#:: Application starts here
#::------------------------


app = QApplication(sys.argv)  # creates the PyQt5 application
mn = Menu()  # Cretes the menu
sys.exit(app.exec_())  # Close the application
