from tkinter import *
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import *


root = Tk()

root.title('Breast cancer prediction')
root.geometry('1200x400')
root.config(bg='lightblue')
header = Label(root,text='Breast cancer predictor',bg='lightblue',foreground='black',font=('Arial',15,'bold')).pack()

frame1= Frame(root,bg='lightblue',padx=10,pady=10)
frame1.pack()

features = ['radius_mean','texture_mean','perimeter_mean','area_mean',
            'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
            'symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
            'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
            'concave points_se','symmetry_se','fractal_dimension_se','smoothness_worst',
            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst',
            'fractal_dimension_worst','radius_worst','texture_worst','perimeter_worst','area_worst']

def create_labels_and_entries(frame,features,start_row=0, start_column=0,bg='lightblue'):
        
        entries = {}
        for idx,feature in enumerate(features):
            row = start_row + idx // 4
            column = start_column + (idx % 4) * 2

            label = Label(frame,text=feature,bg='lightblue',font=('Arial',10,'bold'))
            label.grid(row=row,column=column)

            entry = Entry((frame),bg='white',font=('Arial',10,'bold'))
            entry.grid(row=row,column=column+1)

            entries[feature] = entry

        return entries

#Call the saved model and make predictions
def predict():
          data = []
          for feature in features:
                    data.append(float(entries[feature].get()))
          
          model = joblib.load('logistic_regression.joblib')
          prediction = model.predict([data])
          prediction_proba = model.predict_proba([data])
          confidence = max(prediction_proba[0])
          
          if prediction[0] == 1:
                    result_label.config(text=f'The cancer is malignant with confidence level: {confidence:.2f}', fg='red')
          else:
                    result_label.config(text=f'The cancer is benign with confidence level: {confidence:.2f}', fg='green')

entries = create_labels_and_entries(frame1,features)
predict_button = Button(root,text='Predict',bg='white',font=('Arial',10,'bold'),command=predict)
predict_button.pack()

result_label = Label(root,text='',bg='lightblue',font=('Arial',10,'bold'))
result_label.pack()

root.mainloop()

