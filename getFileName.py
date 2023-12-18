# -*- coding: utf-8 -*-
"""
Created on Sun May 1 14:40:08 2022
"""
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory

#%% Function to get a file name

def getFileName(promptStr, initialDir='your_initial_directory_here', fileTypes=[('All files', '*.*')]):
    """ 
    Opens a dialog box to get a filename.
    
    Parameters:
        - promptStr (str): Dialog header.
        - initialDir (str): Initial directory path (replace it with the actual path).
        - fileTypes (list): List of file types to filter (default is all files).
    
    Returns:
        str: The selected file address.
    """    
    root = tk.Tk()
    root.withdraw()
    fileName = askopenfilename(title=promptStr, initialdir=initialDir, filetypes=fileTypes)
    return fileName

#%% Function to get save file name

def getSaveFileName(promptStr, initialDir='your_initial_directory_here', fileTypes=[('All files', '*.*')]):
    """ 
    Opens a dialog box to get a save filename.
    
    Parameters:
        - promptStr (str): Dialog header.
        - initialDir (str): Initial directory path (replace it with the actual path).
        - fileTypes (list): List of file types to filter (default is all files).
    
    Returns:
        str: The selected file address.
    """ 
    root = tk.Tk()
    root.withdraw()
    fileName = asksaveasfilename(title=promptStr, initialdir=initialDir, filetypes=fileTypes)
    return fileName

#%% Function to get a directory

def getDirectory(promptStr, initialDir='your_initial_directory_here', mustexist=True):
    """ 
    Opens a dialog box to get a directory.
    
    Parameters:
        - promptStr (str): Dialog header.
        - initialDir (str): Initial directory path (replace it with the actual path).
        - mustexist (bool): Whether the selected directory must exist (default is True).
    
    Returns:
        str: The selected directory address.
    """    
    root = tk.Tk()
    root.withdraw()
    directory = askdirectory(title=promptStr, initialdir=initialDir)
    return directory
