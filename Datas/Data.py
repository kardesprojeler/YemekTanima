import pyodbc as odbc
from tkinter import messagebox
from tkinter import filedialog
from tkinter import *
import shutil
import os
import numpy as np
from PIL import Image as pilimage
import tensorflow as ts
import matplotlib.image as mtplot

fields = 'Sınıf İsmi', 'Klasör İsmi'
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=localhost\SQLEXPRESS;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=yes;'
    )

class Data:
    cnxn = odbc.connect(conn_str)
    cursor = cnxn.cursor()
    training_images = []
    training_labels = []

    def get_sinif_list(self):
        array = []
        self.cursor.execute("select * from tbl_01_01_sinif")
        row = self.cursor.fetchone()
        while row:
            sinif = DataSinif()
            sinif.labelnumber = row[0]
            sinif.sinifname = row[1]
            sinif.foldername = row[2]
            array.append(sinif)
            row = self.cursor.fetchone()
        return array

    def onehotlabel(self, sinif, siniflist):
        label = np.zeros((siniflist.__len__(),), dtype=int)
        findingindex = siniflist.index(sinif)
        label[findingindex] = 1
        return label
        pass
    def reshape(self):
        for image in self:

            pass
        pass
    def readimages(self, heigh, width):
        siniflist = self.get_sinif_list()
        for sinif in siniflist:
            path = os.path.join(os.getcwd(), "images",  sinif.foldername)
            for filename in os.listdir(path):
                file_content = pilimage.open(path + "\\" + filename)
                im = file_content.resize((width, heigh), pilimage.ANTIALIAS)
                self.training_labels.append(self.onehotlabel(sinif, siniflist))
                self.training_images.append([np.array(im)])
                pass
            row = Data.cursor.fetchone()
            pass
        #Data.reshape(Data.training_images)
        Data.training_images = np.array([i for i in Data.training_images]).reshape(len(Data.training_images), heigh, width, 3)
        return self.training_images, self.training_labels
        pass
    def insertsinif(cls, sinifname, foldername):
        Data.cursor.execute("select * from tbl_01_01_sinif where (sinifname = ? or " +
                            "foldername = ?)", sinifname, foldername)
        row = Data.cursor.fetchone()
        if row:
            messagebox.showinfo("Uyarı!", "Böyle Bir Satır Mevcut")
            pass
        else:
            Data.cursor.execute("insert into tbl_01_01_sinif(sinifname, foldername) " +
                                "values(?, ?)", sinifname, foldername)
            Data.cnxn.commit()
            pass
        pass
    def getsinifcount(self):
        self.cursor.execute("select sinifcount = count(*) from tbl_01_01_sinif")
        row = Data.cursor.fetchone()
        if row is not None:
            return row[0]
        return 0
        pass
    def adddatasinif(self, entries):
        sinifname = entries[0][1].get()
        foldername = entries[1][1].get()
        if sinifname is not '' and foldername is not '':
            if not os.path.exists(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername):
                os.mkdir(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername)
                Data.insertsinif(entries, foldername, sinifname)
                pass
            else:
                messagebox.showinfo("", "Bu sınıf ismi mevcut!")
                pass
            pass
        else:
            messagebox.showinfo("", "Alanları doldurunuz!")
            pass
        pass

    def adddatasinifx(self):
        root = Tk()
        ents = Data.makeform(root, fields)

        root.bind('<Return>', (lambda event, e=ents: self.adddatasinif(e)))
        b1 = Button(root, text='Tamam',
                    command=(lambda e=ents: self.adddatasinif(e)))
        b1.pack(side=LEFT, padx=5, pady=5)

        b2 = Button(root, text='İptal', command=exit)
        b2.pack(side=LEFT, padx=5, pady=5)
        root.mainloop()
        pass
    def makeform(root, fields):
        entries = []
        for field in fields:
            row = Frame(root)
            lab = Label(row, width=15, text=field, anchor='w')
            ent = Entry(row)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries.append((field, ent))
        return entries

    def addfile(self):
        src = filedialog.askopenfile()
        dest = filedialog.askdirectory(initialdir=r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images')
        if not 1 != 1:
            shutil.copy(src.name, dest)
            pass
        else:
            messagebox.showinfo('Bu dosya mevcut')
        pass
    pass

    def random_batch(self, batch_size, images_size):
        images = []
        labels = []
        for i in range(batch_size):
            random_number = np.random.randint(0, images_size)
            images.append(self.training_images[random_number])
            labels.append(self.training_labels[random_number])
        return images, labels
        pass


class DataSinif:
    labelnumber= -1
    sinifname = ""
    foldername=""
    pass
