# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:28:39 2021

@author: adamkritz
"""

# preprocessing code

import pandas as pd
import os

# you will have to change to your directory
os.chdir(r'C:\Users\trash\Desktop\data 6103 work\movies for project')

# I used my directory for now, we wi

movies = pd.read_csv('IMDb movies.csv')
names = pd.read_csv('IMDb names.csv')
ratings = pd.read_csv('IMDb ratings.csv')
TP = pd.read_csv('IMDb title_principals.csv')

# one value in the year column is mislabeled, this causes a warning when importing
movies['year'] = movies['year'].replace('TV Movie 2019', 2019)

# Merging Ratings and Movies (it merges perfectly)
movies_ratings = movies.merge(ratings, on = 'imdb_title_id')

# Merging movies+ratings with title_principal
# this merges with 9 extra from the movies_ratings side (9 movies are in movies_ratings that arent in title_principal)
# and 19 (actually just 2 repeated a lot) extra movies from the title_principal side (19 movies are in title_principal that are not in movies or ratings)
INDmovies_ratings_TP = movies_ratings.merge(TP, on = 'imdb_title_id', how = 'outer', indicator = True)

# list of the 9 movies
INDmovies_ratings_TP.loc[INDmovies_ratings_TP['_merge'] == 'left_only']['imdb_title_id']
onlyin_MR = ['tt10764458', 'tt11010804', 'tt11777308', 'tt3978706', 'tt4045476', 'tt4045478', 'tt4251266', 'tt5440848', 'tt6889806']

# list of the 2 movies repeated 19 times
INDmovies_ratings_TP.loc[INDmovies_ratings_TP['_merge'] == 'right_only']['imdb_title_id']
onlyin_tp = ['tt1860336'] * 10 + ['tt2082513'] * 9

# merge again without indicator
movies_ratings_TP = movies_ratings.merge(TP, on = 'imdb_title_id', how = 'outer')

# Merges movies+ratings+title_principal with names
INDMRtpN = movies_ratings_TP.merge(names, on = 'imdb_name_id', how = 'outer', indicator = True)

# list of 10 movies only in movies, ratings, and title_principal (9 of them only in movies and ratings, not title_principal)
INDMRtpN.loc[INDMRtpN['_merge'] == 'left_only']['imdb_title_id']
onlyin_MRtp = ['tt0091454', 'tt10764458', 'tt11010804', 'tt11777308', 'tt3978706', 'tt4045476', 'tt4045478', 'tt4251266', 'tt5440848', 'tt6889806']

# final product
MRtpN = movies_ratings_TP.merge(names, on = 'imdb_name_id', how = 'outer')
print(MRtpN)






# creating a mock gui from tutorials

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox 
from PyQt5.QtWidgets import QCheckBox  
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import QSize 
from PyQt5.QtWidgets import QRadioButton 

from PyQt5.QtWidgets import QPushButton # pushbutton 
from PyQt5.QtWidgets import QLineEdit # Lineedit

class CheckControlClass(QMainWindow):
    #send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

   def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(140, 40))    
        self.setWindowTitle("Checkbox") 

        self.b = QCheckBox("Awesome?",self)
        self.b.stateChanged.connect(self.clickBox)
        self.b.move(20,20)
        self.b.resize(320,40)            

   def clickBox(self, state):
        if state == Qt.Checked:
            print('Checked')
        else:
            print('Unchecked')
            
            
class LEditButtonClass(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(LEditButtonClass, self).__init__()

        self.Title = 'Title : Line input and action '
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget) 
        self.exlabel = QLabel("<to be copied here>",self) # exlabel can be use in all the methods in the this class
        self.txtInputText = QLineEdit(self)

        self.btnCopyAction = QPushButton("Copy Text",self)
        self.btnCopyAction.clicked.connect(self.CopyText)

        self.layout.addWidget(self.exlabel)
        self.layout.addWidget(self.txtInputText)
        self.layout.addWidget(self.btnCopyAction)
        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(300, 100)                         # Resize the window


    def CopyText(self):
        self.exlabel.setText(self.txtInputText.text())
        
class RadioButtonClass(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(RadioButtonClass, self).__init__()

        self.Title = 'Title : Radio Button '
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.b1 = QRadioButton("Button 1")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b1)

        self.b2 = QRadioButton("Button 2")
        self.b2.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b2)

        self.b3 = QRadioButton("Button 3")
        self.b3.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b3)

        self.buttonlabel= QLabel('Button 1 is selected',self)
        self.layout.addWidget(self.buttonlabel)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(300, 100)                         # Resize the window


    def onClicked(self):
        button = self.sender()
        if button.isChecked():
            self.buttonlabel.setText(button.text()+' is selected')

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
        
        preproc = mainMenu.addMenu('Preprocessing') 
        
        
        model = mainMenu.addMenu('Modelling') 
        

        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        
        # modelling
        
        model1button = QAction('DT',  self)
        model1button.setStatusTip("Here is our DT")   
        model1button.triggered.connect(self.runcheckbox)  
        
        model2button = QAction('Random Forest',  self)
        model2button.setStatusTip("Here is our RF")   
        model2button.triggered.connect(self.ExampleLEditButton)  
        
        model3button = QAction('NB',  self)
        model3button.setStatusTip("Here is our NB")   
        model3button.triggered.connect(self.ExampleRadioButton)  

        model.addAction(model1button)
        model.addAction(model2button)
        model.addAction(model3button)

        # preprocessing tabs
        
        preproc1Button = QAction("Boxplot", self)   
        preproc1Button.setStatusTip("Here is our Boxplot")   
        preproc1Button.triggered.connect(self.preproc)    

        preproc2Button = QAction("histogram", self)   
        preproc2Button.setStatusTip("Here is our hist")  
        preproc2Button.triggered.connect(self.preproc2)

        preproc.addAction(preproc1Button)    
        preproc.addAction(preproc2Button)
        
        # exit tabs
        
        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)


        # This line shows the windows
        
        self.dialogs = list()

        self.show()
        
    def preproc(self):    # No. 2
        QMessageBox.about(self, "Boxplot", "Boxplot WOW") 
        
    def preproc2(self):
        QMessageBox.about(self, "hist", "hist WOW") 
        
      
    def runcheckbox(self):
        dialog = CheckControlClass()
        self.dialogs.append(dialog)  # Appends the list of dialogs
        dialog.show() 
    
    def ExampleLEditButton(self):
        dialog = LEditButtonClass()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()
        
    def ExampleRadioButton(self):
        dialog = RadioButtonClass()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()


#::------------------------
#:: Application starts here
#::------------------------

def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

if __name__ == '__main__':
    main()

# kaggle has its own package for downloading its stuff
# i used pip and an .ipy file to download it in the script

!pip install kaggle
import pandas as pd
import kaggle 
import os

def download():
    

    # dont push y if you dont wanna make you own working directiory
    a = input('set a working directory to download to? (y/n)')

    if a == 'y':
        b = input('input working directory')
        os.chdir(b)

    # this is just my username and a key they generated for me
    os.environ['KAGGLE_USERNAME'] = "adamkritz"
    os.environ['KAGGLE_KEY'] = "05c2d7615e732897d1f9e6f75613ee41"
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('stefanoleone992/imdb-extensive-dataset', unzip = True)
    directory_path = os.getcwd()

    ratings = pd.read_csv(directory_path + '/IMDb ratings.csv')
    movies = pd.read_csv(directory_path + '/IMDb movies.csv')
    names = pd.read_csv(directory_path + '/IMDb names.csv')
    
    return ratings, movies, names
    
download()

# type ratings or something to see if it works
