import tkinter as tk
from tkinter import filedialog
from Datas import Data

top = tk.Tk()


def AddDataSinif():
    file = filedialog.askdirectory()
    data = Data.Data()
    pass

def AddFile():
    filename = filedialog.asksaveasfilename()
    print(filename)
    return filename
    pass

def hello():
    pass


menubar = tk.Menu(top)
# create a pulldown menu, and add it to the menu bar
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Klasör Ekle", command=AddDataSinif)
filemenu.add_command(label="Dosya Ekle", command=AddFile)
filemenu.add_separator()
filemenu.add_command(label="Çıkış", command=top.quit)
menubar.add_cascade(label="Dosya", menu=filemenu)

# create more pulldown menus
editmenu = tk.Menu(menubar, tearoff=0)
editmenu.add_command(label="Modeli Oluştur", command=hello)
editmenu.add_command(label="Modeli Resetle", command=hello)
menubar.add_cascade(label="Model", menu=editmenu)

helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="Hakkımızda", command=hello)
menubar.add_cascade(label="Yardım", menu=helpmenu)

# display the menu
top.config(menu=menubar)
top.geometry("500x300")
top.title("Yemek Tanıma")
top.mainloop()