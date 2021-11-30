# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:04:12 2021

@author: adamkritz
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import pandas as pd
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication
import webbrowser
from PyQt5.QtWidgets import QSizePolicy


from PyQt5.QtWidgets import QCheckBox    # checkbox
from PyQt5.QtWidgets import QPushButton  # pushbutton
from PyQt5.QtWidgets import QLineEdit    # Lineedit
from PyQt5.QtWidgets import QRadioButton # Radio Buttons
from PyQt5.QtWidgets import QGroupBox    # Group Box

from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

#----------------------------------------------------------------------
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout


# These components are essential for creating the graphics in pqt5 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
from matplotlib.figure import Figure 
import seaborn as sns
from preprocessing_utils import get_repo_root
from preprocessing_utils import get_repo_root_w
from sys import platform
import zipfile


def unzip_results():
    if platform == "darwin":
        with zipfile.ZipFile(get_repo_root() + "/results.zip","r") as zf:
            zf.extractall(get_repo_root())
    elif platform == "win32":
        with zipfile.ZipFile(get_repo_root_w() + "\\results.zip","r") as zf:
            zf.extractall(get_repo_root_w())
       
df = 0
pred = 0

def take(x, y):
    global df
    global pred
    df = x
    pred = y

### The way this works:
### Each class is a separate window. Within each class you can define how you want
### the window to look (size, whats in it, etc)
### The Menu class at the bottom is the main window. In this window there is 
### a file menu that contains the spots for each other window. 
### All the functions at the bottom of the main window will open the other windows
### if they are clicked.

# Numerical Variables Window 
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

        self.groupBox3 = QGroupBox('Graphic 1')
        self.groupBox3Layout = QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        
        self.groupBox4 = QGroupBox('Graphic 2')
        self.groupBox4Layout = QHBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)


        self.b1 = QRadioButton("Duration")
        self.b1.toggled.connect(self.onClicked)


        self.b2 = QRadioButton("Budget")
        self.b2.toggled.connect(self.onClicked)

        self.b3 = QRadioButton("World Wide Gross Income")
        self.b3.toggled.connect(self.onClicked)
        
        self.b4 = QRadioButton("USA Gross Income")
        self.b4.toggled.connect(self.onClicked)
        
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
    
        
        # second graph
        
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        
        #self.toolbar = NavigationToolbar(self.canvas2, self)
        
        self.groupBox3Layout.addWidget(self.canvas)
        
        self.groupBox4Layout.addWidget(self.canvas2)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)
        self.layout.addStretch(1)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()                        # Resize the window
    
    def onClicked(self):
        if self.b1.isChecked():
            mean = str(round(df['duration'].mean()))
            std = str(round(df['duration'].std()))
            self.label.setText('The duration of the movie. The mean is ' + mean + ' minutes and the standard deviation is ' + std + ' minutes.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            sns.histplot(data = df, x = 'duration', ax = self.ax1, kde = True, bins = 75)
            self.ax1.set_xlim((0, 300))
            self.ax1.set_title('Distribution of Movie Durations')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b2.isChecked():
            mean = str(round(df['budget_adjusted'].mean()))
            std = str(round(df['budget_adjusted'].std()))
            self.label.setText('The budget for the movie. This has been adjusted for inflation. The mean is ' + mean + ' USD and the standard deviation is ' + std + ' USD.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            sns.histplot(data = df, x = 'budget_adjusted', ax = self.ax1, kde = True, log_scale = True)
            self.ax1.set_title('Distribution of Movie Budget')
            self.ax1.set_xlabel('Budget ($)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b3.isChecked():
            
            mean = str(round(df['worldwide_gross_income_adjusted'].mean()))
            std = str(round(df['worldwide_gross_income_adjusted'].std()))
            
            self.label.setText('The amount of money the movie made world-wide. This has been adjusted for inflation. The mean is ' + mean + ' USD and the standard deviation is ' + std + ' USD.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.show()
            sns.histplot(data = df, x = 'worldwide_gross_income_adjusted', kde = True, ax = self.ax1, log_scale = True)
            self.ax1.set_title('Distribution of Movie Income (Worldwide)')
            self.ax1.set_xlabel('Worldwide Income ($)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
            
            p = sns.scatterplot(data = df, x = 'date_published_year', y = 'worldwide_gross_income_adjusted', alpha = 0.2, ax = self.ax2)
            p.set(yscale = 'log')
            self.ax2.set_title('Worldwide Gross Income by Date')
            self.ax2.set_ylabel('Worldwide Gross Income ($)')
            self.ax2.set_xlabel('Year Published')
            sns.despine()
            
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()
                
        if self.b4.isChecked():
            
            mean = str(round(df['usa_gross_income_adjusted'].mean()))
            std = str(round(df['usa_gross_income_adjusted'].std()))

            self.label.setText('The amount of money the movie made in the United States. This has been adjusted for inflation. The mean is ' + mean + ' USD and the standard deviation is ' + std + ' USD.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.show()
            sns.histplot(data = df, x = 'usa_gross_income_adjusted', kde = True, ax = self.ax1,  log_scale = True)
            self.ax1.set_title('Distribution of Movie Income (USA)')
            self.ax1.set_xlabel('USA Income ($)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
            p = sns.scatterplot(data = df, x = 'date_published_year', y = 'usa_gross_income_adjusted', alpha = 0.2, ax = self.ax2)
            p.set(yscale = 'log')
            self.ax2.set_title('USA Gross Income by Date')
            self.ax2.set_ylabel('USA Gross Income ($)')
            self.ax2.set_xlabel('Year Published')
            sns.despine()
            
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()

            
# Categorical Variables Window 
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

        self.groupBox3 = QGroupBox('Graphic 1')
        self.groupBox3Layout = QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)# Creates vertical layout
        
        self.groupBox4 = QGroupBox('Graphic 2')
        self.groupBox4Layout = QHBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)

        self.b1 = QRadioButton("Title")
        self.b1.toggled.connect(self.onClicked2)

        self.b2 = QRadioButton("Date Published")
        self.b2.toggled.connect(self.onClicked2)

        self.b3 = QRadioButton("Genre")
        self.b3.toggled.connect(self.onClicked2)
        
        self.b5 = QRadioButton("Region")
        self.b5.toggled.connect(self.onClicked2)
        
        self.b6 = QRadioButton("Director Frequency")
        self.b6.toggled.connect(self.onClicked2)
        
        self.b7 = QRadioButton("Writer Frequency")
        self.b7.toggled.connect(self.onClicked2)
        
        self.b8 = QRadioButton("Production Company Frequency")
        self.b8.toggled.connect(self.onClicked2)
        
        self.b9 = QRadioButton("Actor Frequency")
        self.b9.toggled.connect(self.onClicked2)

        self.b10 = QRadioButton("Description")
        self.b10.toggled.connect(self.onClicked2)
        
        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b10)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.b3)
        self.groupBox1Layout.addWidget(self.b5)
        self.groupBox1Layout.addWidget(self.b6)
        self.groupBox1Layout.addWidget(self.b7)
        self.groupBox1Layout.addWidget(self.b8)
        self.groupBox1Layout.addWidget(self.b9)
        
        self.label = QLabel("")
        self.layout.addWidget(self.label)
        self.groupBox2Layout.addWidget(self.label)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
    
        
        # second graph
        
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        
        #self.toolbar = NavigationToolbar(self.canvas2, self)
        
        self.groupBox3Layout.addWidget(self.canvas)
        
        self.groupBox4Layout.addWidget(self.canvas2)


        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)
        self.layout.addStretch(1)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()                       # Resize the window


    def onClicked2(self):
        if self.b1.isChecked():
            mean = str(round(df['title_n_words'].mean()))
            std = str(round(df['title_n_words'].std()))
            self.label.setText('Having to do with the title of the movie. This includes: number of words in the title, the ratio of long words to short words, the ratio of vowels to not vowels, and the ratio of capital letters to lowercase letters. \nHere are two plot on the number of words and the ratio of long words. For our purposes, these will be treated as categorical variables. The mean number of words in the title is ' + mean +' and the standard deviation is ' + std + '.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.show()
            sns.countplot(data = df, x = 'title_n_words', ax = self.ax1)
            self.ax1.set_title('Number of Words in Title')
            self.ax1.set_xlabel('Number of Words')
            self.ax1.set_ylabel('Frequency')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        
            sns.histplot(data = df, x = 'title_ratio_long_words', ax = self.ax2, bins = 10)
            self.ax2.set_title('Ratio of Long Words to Short Words in the Title')
            self.ax2.set_xlabel('Ratio')
            self.ax2.set_ylabel('Frequency')
            sns.despine()
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()
            
        if self.b10.isChecked():
            mean = str(round(df['description_n_words'].mean()))
            std = str(round(df['description_n_words'].std()))
            self.label.setText('Having to do with the IMDb description of the movie. This is similar to title, and includes: number of words in the description, the ratio of long words to short words, the ratio of vowels to not vowels, and the ratio of capital letters to lowercase letters. \nHere are two plot on the number of words and the ratio of long words. For our purposes, these will be treated as categorical variables. The mean number of words in the description is ' + mean +' and the standard deviation is ' + std + '.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.show()
            sns.countplot(data = df, x = 'description_n_words', ax = self.ax1)
            self.ax1.set_title('Number of Words in Description')
            self.ax1.set_xlabel('Number of Words')
            self.ax1.set_ylabel('Frequency')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        
            sns.histplot(data = df, x = 'description_ratio_long_words', ax = self.ax2, bins = 10)
            self.ax2.set_title('Ratio of Long Words to Short Words in the Description')
            self.ax2.set_xlabel('Ratio')
            self.ax2.set_ylabel('Frequency')
            sns.despine()
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()
            
        if self.b2.isChecked():
            self.label.setText('The date the movie released. This includes the year, month, and day of release. Here are two plots on the year and month of release.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.show()
            sns.histplot(data = df, x = 'date_published_year', ax = self.ax1)
            self.ax1.set_title('Year Released')
            self.ax1.set_xlabel('Year')
            self.ax1.set_ylabel('Frequency')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
            sns.countplot(data = df, x = 'date_published_month', ax = self.ax2)
            self.ax2.set_title('Month Released')
            self.ax2.set_xlabel('Month')
            self.ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',' Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            self.ax2.set_ylabel('Frequency')
            sns.despine()
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()
            
        if self.b3.isChecked():
            self.label.setText('The genre of the movie. IMDb gives each movie 3 genres, resulting in many possible genre combinations. There are also many different genres that movies can have, further increasing the number of genres. \nBecause of this, we decided to binary encode the genres, as we could not easily represent each one otherwise. This results in 732 combinations of genres that we will use.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.hide()
            self.groupBox4.hide()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b5.isChecked():
            self.label.setText('The region the movie was initially released in. The six regions are: Africa, Americas, Asia, Europe, Oceania, and None of the above/No region recorded.')
            af = df['region_Africa'].value_counts()[1]
            am = df['region_Americas'].value_counts()[1]
            asi = df['region_Asia'].value_counts()[1]
            eu = df['region_Europe'].value_counts()[1]
            oc = df['region_Oceania'].value_counts()[1]
            no = df['region_None'].value_counts()[1]
            vals = [af, am, asi, eu, oc, no]
            names = ['Africa', 'America', 'Asia', 'Europe', 'Oceania', 'None']
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            self.ax1.bar(names, height = vals)
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b6.isChecked():
            mean = str(round(df['director_weighted_frequency'].mean()))
            std = str(round(df['director_weighted_frequency'].std()))
            self.label.setText('The frequency of director appearance. This variable measures how often a director directs a movie compared to other directors. For our purposes, this will be represented as a categorical variable. The mean frequency is ' + mean + ' and the standard deviation is ' + std + '.')
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            sns.histplot(data = df, x = 'director_weighted_frequency', ax = self.ax1, bins = 20)
            self.ax1.set_title('Director Frequency Histogram')
            self.ax1.set_xlabel('Directory Frequency')
            self.ax1.set_ylabel('Frequency (Count)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b7.isChecked():
            mean = str(round(df['writer_weighted_frequency'].mean()))
            std = str(round(df['writer_weighted_frequency'].std()))
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            self.label.setText('The frequency of writer appearance. This variable measures how often a writer writes a movie compared to other writer. For our purposes, this will be represented as a categorical variable. The mean frequency is ' + mean + ' and the standard deviation is ' + std + '.')
            sns.histplot(data = df, x = 'writer_weighted_frequency', ax = self.ax1, bins = 20)
            self.ax1.set_title('Writer Frequency Histogram')
            self.ax1.set_xlabel('Writer Frequency')
            self.ax1.set_ylabel('Frequency (Count)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b8.isChecked():
            mean = str(round(df['production_company_frequency'].mean()))
            std = str(round(df['production_company_frequency'].std()))
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            self.label.setText('The frequency of production company appearance. This variable measures how often a production company produces a movie compared to other production companies. For our purposes, this will be represented as a categorical variable. The mean frequency is ' + mean + ' and the standard deviation is ' + std + '.')
            sns.histplot(data = df, x = 'production_company_frequency', ax = self.ax1, bins = 20)
            self.ax1.set_title('Production Company Frequency Histogram')
            self.ax1.set_xlabel('Production Company Frequency')
            self.ax1.set_ylabel('Frequency (Count)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            
        if self.b9.isChecked():
            mean = str(round(df['actors_weighted_frequency'].mean()))
            std = str(round(df['actors_weighted_frequency'].std()))
            self.ax1.clear()
            self.ax2.clear()
            self.groupBox3.show()
            self.groupBox4.hide()
            self.label.setText('The frequency of actor appearance. This variable measures how often a actor acts in a movie compared to other actors. For our purposes, this will be represented as a categorical variable. The mean frequency is ' + mean + ' and the standard deviation is ' + std + '.')
            sns.histplot(data = df, x = 'actors_weighted_frequency', ax = self.ax1, bins = 20)
            self.ax1.set_title('Actor Frequency Histogram')
            self.ax1.set_xlabel('Actor Company Frequency')
            self.ax1.set_ylabel('Frequency (Count)')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        

# Target Variable Window            
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
        
        self.groupBox2 = QGroupBox('Plot Picker')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.groupBox25 = QGroupBox('Navigation Bar')
        self.groupBox25Layout= QHBoxLayout()
        self.groupBox25.setLayout(self.groupBox25Layout)
        
        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout= QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        
        mean = str(round(df['weighted_average_vote'].mean()))
        std = str(round(df['weighted_average_vote'].std()))
        
        self.label = QLabel("The average vote for an IMDb movie is calculated by the averaging all the ratings for a movie. However, IMDb uses weighted average vote over raw average. \nThis allows IMDb to weight votes differently in order to detect unusual activity, like review-bombing. This allows IMDb to prevent users from drastically changing a movie's score. \nThe mean weighted average vote is " + mean + ' and the standard deviation is ' +std + '. This will be our target variable to predict.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.fig = Figure()
        gs00 = grd.GridSpec(1, 2, width_ratios=[10,1])
        self.ax1 = self.fig.add_subplot(gs00[0])
        self.cax = self.fig.add_subplot(gs00[1])
        self.canvas = FigureCanvas(self.fig)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.groupBox25Layout.addWidget(self.toolbar)
        
        self.b1 = QRadioButton("Distribution")
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("Heatmap")
        self.b2.toggled.connect(self.onClicked)
        
        self.groupBox2Layout.addWidget(self.b1)
        self.groupBox2Layout.addWidget(self.b2)
        

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        
        self.groupBox3Layout.addWidget(self.canvas)

        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox25)
        self.layout.addWidget(self.groupBox3)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()                      # Resize the window

    def onClicked(self):
        if self.b1.isChecked():
            self.ax1.clear()
            self.cax.set_visible(False)
            sns.histplot(data = df, x = 'weighted_average_vote', ax = self.ax1, bins = 40, kde = True)
            self.ax1.set_title('Distribution of Weighted Votes')
            sns.despine()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
        if self.b2.isChecked():
            self.ax1.clear()
            self.cax.set_visible(True)
            sns.heatmap(df[['duration', 'weighted_average_vote', 'budget_adjusted',
                    'usa_gross_income_adjusted', 'worldwide_gross_income_adjusted',
                    'date_published_year', 'date_published_month', 'date_published_day',
                    'actors_weighted_frequency', 'director_weighted_frequency',
                    'writer_weighted_frequency', 'production_company_frequency', 'title_n_words',
                    'title_ratio_long_words', 'title_ratio_vowels',
                    'title_ratio_capital_letters', 'description_n_words',
                    'description_ratio_long_words', 'description_ratio_vowels',
                    'description_ratio_capital_letters', ]].corr(), vmin = -1, vmax = 1, ax = self.ax1, cmap = 'coolwarm', cbar_ax=self.cax)
            self.ax1.set_title('Correlation Matrix of Numeric Variables')
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()


class ModelstoTry(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(ModelstoTry, self).__init__()
        
        self.Title = 'Models to Try'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget) 
        
        self.groupBox1 = QGroupBox('Background')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel('We started our modeling phase by first selecting a handful of promising models that work with regression problems.\
                            \nOur selected models are Linear Regression, Random Forest, Gradient Boosting, Adaptive Boosting, and K-Nearest Neighbors.\
                            \nThe out of the box Random Forest and Gradient Boosting models seem to perform best with the 2 lowest validation MSEs.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.groupBox2 = QGroupBox('MSE Plot')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.label_image = QLabel()
        if platform == "darwin":
            self.pix = QPixmap(get_repo_root() + '/results/1. Base Model Comparison/model_comparison_base_all_features.png')
            self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
        elif platform == "win32":
            self.pix = QPixmap(get_repo_root_w() + '\\results\\1. Base Model Comparison\\model_comparison_base_all_features.png')
            self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
        self.groupBox2Layout.addWidget(self.label_image)
        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()    
        
class Hyp1(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(Hyp1, self).__init__()
        
        self.Title = 'Hyperparameter Tuning and Validation Phase I'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget) 
        
        self.groupBox1 = QGroupBox('Info')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel('We selected the top 3 base models and tuned them by setting a list of \
                            \nhyperparameters to try in GridSearch validation to see if performance increases.\
                            \nAs seen in the Model Comparison graph, our best model still seems to be the \
                            \nRandom Forest. Let us focus on just that model in the next phase.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.groupBox15 = QGroupBox('Plot Picker')
        self.groupBox15Layout= QHBoxLayout()
        self.groupBox15.setLayout(self.groupBox15Layout)
        
        self.b1 = QRadioButton("Random Forest")
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("KNN")
        self.b2.toggled.connect(self.onClicked)

        self.b3 = QRadioButton("Gradient Boosting")
        self.b3.toggled.connect(self.onClicked)
        
        self.b4 = QRadioButton("Model Comparison")
        self.b4.toggled.connect(self.onClicked)

        self.groupBox15Layout.addWidget(self.b4)
        self.groupBox15Layout.addWidget(self.b1)
        self.groupBox15Layout.addWidget(self.b2)
        self.groupBox15Layout.addWidget(self.b3)

        
        self.groupBox2 = QGroupBox('Tuned Plots')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.label_image = QLabel()
        self.groupBox2Layout.addWidget(self.label_image)
        
        
        self.label_image2 = QLabel()
        
        if platform == "darwin":
            self.all_data = pd.read_csv(get_repo_root() + '/results/2. Tuning 1/gridsearchcv_results.csv')
        elif platform == "win32":   
            self.all_data = pd.read_csv(get_repo_root_w() + '\\results\\2. Tuning 1\\gridsearchcv_results.csv')
    
        
        self.groupBox4 = QGroupBox('Gridsearch Results')
        self.groupBox4Layout= QHBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)
    
        NumRows = len(self.all_data.index)
        
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(len(self.all_data.columns))
        self.tableWidget.setRowCount(NumRows)
        self.tableWidget.setHorizontalHeaderLabels(self.all_data.columns)

        for i in range(NumRows):
            for j in range(len(self.all_data.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_data.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        
        self.groupBox4Layout.addWidget(self.tableWidget)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox15, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 1)
        self.layout.addWidget(self.groupBox4, 1, 0)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()   
        
    def onClicked(self):
            if self.b1.isChecked():
                if platform == "darwin":
                    self.pix = QPixmap(get_repo_root() + '/results/2. Tuning 1/validation_curves_random_forest_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                elif platform == "win32":
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\2. Tuning 1\\validation_curves_random_forest_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)

            if self.b2.isChecked():
                if platform == "darwin":
                    self.pix = QPixmap(get_repo_root() + '/results/2. Tuning 1/validation_curves_knn_regressor_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                elif platform == "win32":
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\2. Tuning 1\\validation_curves_knn_regressor_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                    
            if self.b3.isChecked():
                if platform == "darwin":
                    self.pix = QPixmap(get_repo_root() + '/results/2. Tuning 1/validation_curves_gradient_boost_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                elif platform == "win32":
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\2. Tuning 1\\validation_curves_gradient_boost_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                    
                    
            if self.b4.isChecked():
                if platform == "darwin":
                    self.pix = QPixmap(get_repo_root() + '/results/2. Tuning 1/model_comparison_tuning_model1.png')
                    self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                elif platform == "win32":   
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\2. Tuning 1\\model_comparison_tuning_model1.png')
                    self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                  
                    
class Hyp2(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(Hyp2, self).__init__()
        
        self.Title = 'Hyperparameter Tuning and Validation Phase II'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        
        self.groupBox1 = QGroupBox('Info')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel('We tried to tune our best model, the Random Forest model to see how much better we can make it. Our Random Forest showed signs of overfitting so we tried to set hyperparameters to regularize the model \
                            \nsuch as increasing the number of trees, the min samples per leaf node, the max number of features per tree, and the max depth each tree can go.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.b1 = QRadioButton('Learning Curves')
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton('Validation Curves')
        self.b2.toggled.connect(self.onClicked)
        
        self.groupBox15 = QGroupBox('Plot Picker')
        self.groupBox15Layout= QHBoxLayout()
        self.groupBox15.setLayout(self.groupBox15Layout)
        
        self.groupBox15Layout.addWidget(self.b1)
        self.groupBox15Layout.addWidget(self.b2)
        
        self.groupBox2 = QGroupBox('Curve Plots')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.label_image = QLabel()
        
        self.groupBox2Layout.addWidget(self.label_image)
        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox15)
        self.layout.addWidget(self.groupBox2)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()


    def onClicked(self):
            if self.b1.isChecked():
                if platform == "darwin":
                    self.pix = QPixmap(get_repo_root() + '/results/3. Tuning 2 & Model Selection/learning_curves_random_forest_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                elif platform == "win32":
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\3. Tuning 2 & Model Selection\\learning_curves_random_forest_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)
                    
            if self.b2.isChecked():
                if platform == "darwin":
                     self.pix = QPixmap(get_repo_root() + '/results/3. Tuning 2 & Model Selection/validation_curves_random_forest_tuned.png')
                     self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                     self.label_image.setPixmap(self.pix2)
                elif platform == "win32":
                    self.pix = QPixmap(get_repo_root_w() + '\\results\\3. Tuning 2 & Model Selection\\validation_curves_random_forest_tuned.png')
                    self.pix2 = self.pix.scaled(1200, 800, transformMode=Qt.SmoothTransformation)
                    self.label_image.setPixmap(self.pix2)

class ModelSelection(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(ModelSelection, self).__init__()
        
        self.Title = 'Model Selection'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget) 
        
        self.groupBox1 = QGroupBox('Info')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel('This is our final selected model.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.groupBox2 = QGroupBox('Gridsearch Results 2')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.label_image = QLabel()
        
        if platform == "darwin":
            self.pix = QPixmap(get_repo_root() + '/results/4. Results Evaluation/most_important_features_results_eval.png')
            self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
            self.all_data = pd.read_csv(get_repo_root() + '/results/4. Results Evaluation/gridsearchcv_results.csv')
        elif platform == "win32":   
            self.pix = QPixmap(get_repo_root_w() + '\\results\\4. Results Evaluation\\most_important_features_results_eval.png')
            self.pix2 = self.pix.scaled(1000, 500, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
            self.all_data = pd.read_csv(get_repo_root_w() + '\\results\\4. Results Evaluation\\gridsearchcv_results.csv')
            
        NumRows = len(self.all_data.index)   
        
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(len(self.all_data.columns))
        self.tableWidget.setRowCount(NumRows)
        self.tableWidget.setHorizontalHeaderLabels(self.all_data.columns)

        for i in range(NumRows):
            for j in range(len(self.all_data.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_data.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        
        self.groupBox2Layout.addWidget(self.tableWidget)
        
        self.groupBox3 = QGroupBox('Most Important Features')
        self.groupBox3Layout= QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        
        
        self.groupBox3Layout.addWidget(self.label_image)
        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()  


class ModelResults(QMainWindow):

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(ModelResults, self).__init__()
        
        self.Title = 'Model and Results Evaluation'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  
        
        self.groupBox1 = QGroupBox('Info')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel('After selecting our best Random Forest model, we compared it with a random model and calculated the average MSE between the two. Our model performed much better than the random model and was proved statistically significant using a 2-Sample T-Test with a null hypothesis that the two distributions are the same.')
        self.groupBox1Layout.addWidget(self.label)
        
        self.groupBox2 = QGroupBox('Our Model')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        
        self.groupBox3 = QGroupBox('Model vs Random')
        self.groupBox3Layout= QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        
        self.label2 = QLabel('Our model can predict the weighted average movie IMDB rating with an average error of +- 0.93 while a random model has an average error of +- 2.9.')
        
        self.groupBox4 = QGroupBox('Prediction Results')
        self.groupBox4Layout= QHBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)
        
        self.groupBox5 = QGroupBox('MSE: Our Model Versus Random')
        self.groupBox5Layout= QHBoxLayout()
        self.groupBox5.setLayout(self.groupBox5Layout)
        
        self.label_image = QLabel()
        
        if platform == "darwin":
            self.pix = QPixmap(get_repo_root() + '/results/4. Results Evaluation/vs_random_results_eval.png')
            self.pix2 = self.pix.scaled(720,360, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
            self.all_data = pd.read_csv(get_repo_root() + '/results/4. Results Evaluation/best_model_evaluation_results.csv')
            self.all_data2 = pd.read_csv(get_repo_root() + '/results/4. Results Evaluation/prediction_results.csv')
        elif platform == "win32":   
            self.pix = QPixmap(get_repo_root_w() + '\\results\\4. Results Evaluation\\vs_random_results_eval.png')
            self.pix2 = self.pix.scaled(720,360, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
            self.all_data = pd.read_csv(get_repo_root_w() + '\\results\\4. Results Evaluation\\best_model_evaluation_results.csv')
            self.all_data2 = pd.read_csv(get_repo_root_w() + '\\results\\4. Results Evaluation\\prediction_results.csv')
            
        self.all_dataHead = self.all_data.head(6) 
            
        NumRows = len(self.all_dataHead.index)
        
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(len(self.all_dataHead.columns))
        self.tableWidget.setRowCount(NumRows)
        self.tableWidget.setHorizontalHeaderLabels(self.all_dataHead.columns)

        for i in range(NumRows):
            for j in range(len(self.all_dataHead.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_dataHead.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        
        
        self.all_data2Head = self.all_data2.head(10) 
            
        NumRows = len(self.all_data2Head.index)
        
        self.tableWidget2 = QTableWidget()
        self.tableWidget2.setColumnCount(len(self.all_data2Head.columns))
        self.tableWidget2.setRowCount(NumRows)
        self.tableWidget2.setHorizontalHeaderLabels(self.all_data2Head.columns)

        for i in range(NumRows):
            for j in range(len(self.all_data2Head.columns)):
                self.tableWidget2.setItem(i, j, QTableWidgetItem(str(self.all_data2Head.iat[i, j])))

        self.tableWidget2.resizeColumnsToContents()
        self.tableWidget2.resizeRowsToContents()
            
        self.groupBox2Layout.addWidget(self.tableWidget)    
        self.groupBox3Layout.addWidget(self.label2)
        self.groupBox4Layout.addWidget(self.tableWidget2)  
        self.groupBox5Layout.addWidget(self.label_image)    
        
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)
        self.layout.addWidget(self.groupBox5)
        
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()    

# Prediction window
class predi(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication
    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(predi, self).__init__()
        
        self.Title = 'Prediction Game'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout
        
        
        
        self.groupBox1 = QGroupBox('Description')
        self.groupBox1Layout= QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        
        self.label = QLabel("This tool will allow you to make predictions against our best model, to see who can come out on top!\nWe will give you a list of features of a movie selected randomly from our test set and you will predict the weighted average score.\nOur model will also predict the weighted average score, and whoever comes the closest to the real score will win! \nSince you are presumably a human, we will give you human readable features for you to make your guess. \nRemember, no cheating by looking up the movie online. And if any of the features are missing, it is because they were not in the IMDb dataset, so our model did not get them either.")
        self.groupBox1Layout.addWidget(self.label)
        
        self.groupBox15 = QGroupBox('Random Movie Generator')
        self.groupBox15Layout= QHBoxLayout()
        self.groupBox15.setLayout(self.groupBox15Layout)
        
        self.button = QPushButton('Generate', self)
        self.button.setToolTip('This is an example button')
        self.groupBox15Layout.addWidget(self.button)
        self.button.clicked.connect(self.on_click)
        
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(2)
        self.tableWidget.setColumnCount(15)
        self.tableWidget.setItem(0, 0, QTableWidgetItem("Duration"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("Title"))
        self.tableWidget.setItem(0, 2, QTableWidgetItem("Date Published"))
        self.tableWidget.setItem(0, 3, QTableWidgetItem("Director"))
        self.tableWidget.setItem(0, 4, QTableWidgetItem("Writer"))
        self.tableWidget.setItem(0, 5, QTableWidgetItem("Production Company"))
        self.tableWidget.setItem(0, 6, QTableWidgetItem("Actors"))
        self.tableWidget.setItem(0, 7, QTableWidgetItem("Description"))
        self.tableWidget.setItem(0, 8, QTableWidgetItem("Budget"))
        self.tableWidget.setItem(0, 9, QTableWidgetItem("USA Gross Income"))
        self.tableWidget.setItem(0, 10, QTableWidgetItem("Worldwide Gross Income"))
        self.tableWidget.setItem(0, 11, QTableWidgetItem("Genre 1"))
        self.tableWidget.setItem(0, 12, QTableWidgetItem("Genre 2"))
        self.tableWidget.setItem(0, 13, QTableWidgetItem("Genre 3"))
        self.tableWidget.setItem(0, 14, QTableWidgetItem("Region"))
        
        self.groupBox175 = QGroupBox("Your Movie's 'Features")
        self.groupBox175Layout= QHBoxLayout()
        self.groupBox175.setLayout(self.groupBox175Layout)
        self.groupBox175Layout.addWidget(self.tableWidget)
        
        self.groupBox2 = QGroupBox('Input your guess')
        self.groupBox2Layout= QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.txtInputText = QLineEdit(self)

        self.locked = QPushButton("Lock In!",self)
        self.locked.clicked.connect(self.guess)

        self.groupBox2Layout.addWidget(self.txtInputText)
        self.groupBox2Layout.addWidget(self.locked)
        
        self.groupBox3 = QGroupBox("Results")
        self.groupBox3Layout= QHBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        
        self.label3 = QLabel('')
        self.groupBox3Layout.addWidget(self.label3)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox15)
        self.layout.addWidget(self.groupBox175)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.showMaximized()                        # Resize the window

    def on_click(self):
        global movie
        movie = pred.sample(n = 1)
        movie2 = movie[['duration', 'title',
                        'date_published', 'director', 'writer', 'production_company',
                        'actors', 'description', 'budget_adjusted',
                        'usa_gross_income_adjusted', 'worldwide_gross_income_adjusted',
                        'genre1', 'genre2', 'genre3', 'region']]
        movie3 = movie2.to_numpy()
        movie4 = movie3[0]
        self.tableWidget.setItem(1,0, QTableWidgetItem(str(movie4[0])))
        self.tableWidget.setItem(1,1, QTableWidgetItem(str(movie4[1])))
        self.tableWidget.setItem(1,2, QTableWidgetItem(str(movie4[2])))
        self.tableWidget.setItem(1,3, QTableWidgetItem(str(movie4[3])))
        self.tableWidget.setItem(1,4, QTableWidgetItem(str(movie4[4])))
        self.tableWidget.setItem(1,5, QTableWidgetItem(str(movie4[5])))
        self.tableWidget.setItem(1,6, QTableWidgetItem(str(movie4[6])))
        self.tableWidget.setItem(1,7, QTableWidgetItem(str(movie4[7])))
        self.tableWidget.setItem(1,8, QTableWidgetItem(str(movie4[8])))
        self.tableWidget.setItem(1,9, QTableWidgetItem(str(movie4[9])))
        self.tableWidget.setItem(1,10, QTableWidgetItem(str(movie4[10])))
        self.tableWidget.setItem(1,11, QTableWidgetItem(str(movie4[11])))
        self.tableWidget.setItem(1,12, QTableWidgetItem(str(movie4[12])))
        self.tableWidget.setItem(1,13, QTableWidgetItem(str(movie4[13])))
        self.tableWidget.setItem(1,14, QTableWidgetItem(str(movie4[14])))


    def guess(self):
        movie22 = movie[['Actual Rating', 'Predicted Rating']]
        movie23 = movie22.to_numpy()
        movie24 = movie23[0]
        a = self.txtInputText.text()
        b = movie24[1]
        c = movie24[0]
        if abs(c - float(a)) < abs(c - b):    
            self.label3.setText("The results are in...\nYou Predicted: " + a + "\nOur model predicted: " + str(b) + "\nThe actual weighted average vote is: " + str(c) + "\nYou win!")
        if abs(c - float(a)) > abs(c - b):   
            self.label3.setText("The results are in...\nYou Predicted: " + a + "\nOur model predicted: " + str(b) + "\nThe actual weighted average vote is: " + str(c) +"\nYou lose!")
        if abs(c - float(a)) == abs(c - b):   
            self.label3.setText("The results are in...\nYou Predicted: " + a + "\nOur model predicted: " + str(b) + "\nThe actual weighted average vote is: " + str(c) +"\nIt's a tie!")

# Main Menu window
class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        # set size
        self.left = 400
        self.top = 200
        self.width = 1000
        self.height = 700

        # Title

        self.Title = 'Group 2 Final Project'

        #call intiUI to create elements for menu

        self.initUI()
        
    def initUI(self):

        
        # Creates the menu and the items
       
        
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        
        self.main_widget = QWidget(self)
        
        self.label = QLabel(self)
        self.label.setText("Welcome to our final project! \nWe will be using modelling to predict the IMDb score of movies. \nPlease click any of the tabs above to look around.")
        self.label.setFont(QFont("Times", 20))
        self.label.adjustSize()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.move(40, 100)
        
        self.label_image = QLabel(self)
        
        if platform == "darwin":
            self.pix = QPixmap(get_repo_root() + '/adam-kritz-individual-project/IMDb-Logo.png')
            self.pix2 = self.pix.scaled(709, 341, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
        elif platform == "win32":   
            self.pix = QPixmap(get_repo_root_w() + '\\adam-kritz-individual-project\\IMDb-Logo.png')
            self.pix2 = self.pix.scaled(709, 341, transformMode=Qt.SmoothTransformation)
            self.label_image.setPixmap(self.pix2)
            
        self.label_image.adjustSize()
        self.label_image.move(145, 300)
        
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        
        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('File')
        
        preproc = mainMenu.addMenu('EDA') 
        
        model = mainMenu.addMenu('Modelling') 
        
        pred = mainMenu.addMenu('Prediction Game') 
        

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
        
        file3Button = QAction("About Us", self)   
        file3Button.setStatusTip("Information about our project")   
        file3Button.triggered.connect(self.file3)     
    
        
        file5Button = QAction("IMDb Legal Information", self)   
        file5Button.setStatusTip("IMDb Legal Information")   
        file5Button.triggered.connect(self.file5) 
        
        
        file2Button = QAction("Link to our report", self)   
        file2Button.setStatusTip("Here you can find the full report of our results")   
        file2Button.triggered.connect(self.file2)    
        
        
        file4Button = QAction("Link to the dataset", self)   
        file4Button.setStatusTip("Link to the dataset on Kaggle")   
        file4Button.triggered.connect(self.file4) 
        
        
        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        
        
        fileMenu.addAction(file3Button)
        fileMenu.addAction(file5Button)
        fileMenu.addAction(file2Button)
        fileMenu.addAction(file4Button)
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
        
        model1button = QAction('Models to Try',  self)
        model1button.setStatusTip("Models to Try")   
        model1button.triggered.connect(self.model1)  
        
        model2button = QAction('Hyperparameter Tuning and Validation Phase I',  self)
        model2button.setStatusTip("Hyperparameter Tuning and Validation Phase I")   
        model2button.triggered.connect(self.model2)  
        
        model3button = QAction('Hyperparameter Tuning and Validation Phase II',  self)
        model3button.setStatusTip("Hyperparameter Tuning and Validation Phase II")   
        model3button.triggered.connect(self.model3)  
        
        model4button = QAction('Model Selection',  self)
        model4button.setStatusTip("Model Selection")   
        model4button.triggered.connect(self.model4)  
        
        model5button = QAction('Model and Results Evaluation',  self)
        model5button.setStatusTip("Model and Results Evaluation")   
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
    
    def file2(self):
        webbrowser.open('https://docs.google.com/document/d/15mzM34VmwNzyYF0Mygbi-N_v0qPVQXY8LAIWD_Nz-p8/edit?usp=sharing') # this will be our report
        
    def file3(self):
        QMessageBox.about(self, "About Us", "We created this project in fall 2021 as part of our Intro to Data Mining Course at George Washington University. In this project, we took Stefano Leone’s IMDb dataset on Kaggle, and used different modeling techniques to predict the weighted average vote of movies based on their features. ")
    
    def file4(self):
        webbrowser.open('https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset')
    
    def file5(self):
        QMessageBox.about(self, "IMDb License", "IMDb, IMDb.COM, and the IMDb logo are trademarks of IMDb.com, Inc. or its affiliates.")
    
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
        dialog = ModelstoTry()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model2(self):
        dialog = Hyp1()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model3(self):
        dialog = Hyp2()
        self.dialogs.append(dialog) 
        dialog.show()
    
    def model4(self):
        dialog = ModelSelection()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def model5(self):
        dialog = ModelResults()
        self.dialogs.append(dialog) 
        dialog.show()
        
    def pred1(self):
        dialog = predi()
        self.dialogs.append(dialog) 
        dialog.show()
      

#::------------------------
#:: Application starts here
#::------------------------

if __name__ == "__main__":
    unzip_results()
    # change these to your file paths if you want to run the GUI by itself
    df = pd.read_csv(r'C:\Users\trash\Desktop\data 6103 work\moviesdf.csv')
    pred = pd.read_csv(r'C:\Users\trash\Desktop\data 6103 work\predictions_with_ids.csv')
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Creates the menu
    sys.exit(app.exec_())