# Creating GUI for Image Classification

import tkinter as tk
from tkinter import filedialog
from tkinter import*
from PIL import ImageTk, Image
import numpy

#Loading the trained model
from keras.models import load_model
model = load_model('FinalModel.h5')

#dictionary to label dataset
classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
}

#GUI Initialization
top=tk.Tk()
top.geometry('700x700')
top.title('Image Classification using CIFAR10')
top.configure(background='sky blue',borderwidth=5)
label=Label(top,background='#CDCDCD', font=('algerian',15,'bold'))
sign_image = Label(top)

#defining function for running the classification
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    predict_X = model.predict(image)
    classes_X=numpy.argmax(predict_X,axis=1)
    sign = classes[classes_X[0]]
    print(sign)
    label.configure(foreground='black', text=sign, relief='sunken', font=('callibri',
                    30,'bold'))
    label.place(x=240,y=120)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        label.place(x=160,y=100)
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload Image",relief='sunken',command=upload_image,padx=20,pady=10)
upload.configure(background='white',font=('callibri',15,'bold'))
upload.place(x=250,y=600)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="Image Classification",relief='sunken')
heading.config(font=('callibri',25,'bold'))
heading.place(x=160,y=45)
heading.pack()
top.mainloop()
