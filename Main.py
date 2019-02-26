import tkinter as tk
from Models import Model as model
from Datas import Data as data


class Main:
    data = None

    def __init__(self):
        self.data = data.Data()

    def make_main_form(self):
        top = tk.Tk()
        top.maxsize()

        menubar = tk.Menu(top)
        # create a pulldown menu, and add it to the menu bar
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Sınıf Ekle", command=self.add_datasinif)
        filemenu.add_command(label="Dosya Ekle", command=self.data.addfile)
        filemenu.add_separator()
        filemenu.add_command(label="Çıkış", command=top.quit)
        menubar.add_cascade(label="Dosya", menu=filemenu)

        # create more pulldown menus
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Modeli Oluştur", command=self.make_model)
        editmenu.add_command(label="Modeli Eğit", command=self.train_model)
        editmenu.add_command(label="Modeli Test Et", command=self.test_model)
        menubar.add_cascade(label="Model", menu=editmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Hakkımızda", command=None)
        menubar.add_cascade(label="Yardım", menu=helpmenu)

        # display the menu
        top.config(menu=menubar)
        top.geometry("1000x1000")
        top.title("Yemek Tanıma")
        top.mainloop()

    def make_model(self):
        self.model = model.Model()
        self.model.make_model()
        print("Model Oluşturuldu")

    def train_model(self):
        self.model.train_step(5000)

    def test_model(self):
        self.model.test_accuracy()
        pass

    def add_datasinif(self):
        self.data.adddatasinifx()

    def add_file(self):
        self.data.addfile()


if __name__ == '__main__':
    main = Main()
    main.make_main_form()