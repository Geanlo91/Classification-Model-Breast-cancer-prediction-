from tkinter import *
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import *


root = Tk()

root.title('Breast cancer prediction')
root.geometry (600*600)
root.config(bg='lightblue')
header = Label(root,text='Breast cancer predictor',bg='lightblue',foreground='black',font=('Arial',15,'bold'))

frame1= Frame(root,bg='lightblue')
frame1.pack()

Label(fram




root.mainloop()