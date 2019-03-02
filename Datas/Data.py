import pyodbc as odbc
from tkinter import messagebox, filedialog, ttk
import shutil
import os
import numpy as np
from PIL import Image as pilimage
import tensorflow as tf
import wx

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

    test_images = []
    test_labels = []

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
        finding_index = siniflist.index(sinif)
        label[finding_index] = 1
        return label
        pass

    def read_train_images(self, heigh, width):
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
        Data.training_images = np.array([i for i in Data.training_images]).reshape(len(Data.training_images), heigh,
                                                                                   width, 3)
        return self.training_images, self.training_labels
        pass

    def read_test_images(self, heigh, width):
        siniflist = self.get_sinif_list()
        for sinif in siniflist:
            self.cursor.execute('select labelnumber, foldername, filename from tbl_01_01_'
                                'testimage a where labelnumber = ?', sinif.labelnumber)
            row = self.cursor.fetchone()
            while row:
                path = os.path.join(os.getcwd(), "images", row[1], row[2])
                if os.path.exists(path):
                    file_content = pilimage.open(path)
                    im = file_content.resize((width, heigh), pilimage.ANTIALIAS)
                    self.test_labels.append(self.onehotlabel(sinif, siniflist))
                    self.test_images.append([np.array(im)])
                row = Data.cursor.fetchone()
            pass
        self.test_images = np.array([i for i in self.test_images]).reshape(len(self.test_images), heigh,
                                                                           width, 3)
        return self.test_images, self.test_labels

    def insertsinif(self, sinifname, foldername):
        self.cursor.execute("select * from tbl_01_01_sinif where (sinifname = ? or " +
                            "foldername = ?)", sinifname, foldername)
        row = self.cursor.fetchone()
        if row:
            wx.MessageBox('Böyle Bir Satır Mevcut', 'Attention', wx.OK | wx.ICON_WARNING)
            pass
        else:
            self.cursor.execute("insert into tbl_01_01_sinif(sinifname, foldername) " +
                                "values(?, ?)", sinifname, foldername)
            self.cnxn.commit()
            pass
        pass
    def getsinifcount(self):
        self.cursor.execute("select sinifcount = count(*) from tbl_01_01_sinif")
        row = Data.cursor.fetchone()
        if row is not None:
            return row[0]
        return 0
        pass

    def add_data_sinif(self, sinifname, foldername):
        if sinifname is not '' and foldername is not '':
            if not os.path.exists(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername):
                os.mkdir(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername)
                self.insertsinif(foldername, sinifname)
                pass
            else:
                wx.MessageBox('Bu sınıf ismi mevcut', 'Attention', wx.OK | wx.ICON_WARNING)
                pass
            pass
        else:
            wx.MessageBox('Alanları doldurunuz!', 'Attention', wx.OK | wx.ICON_WARNING)
            pass
        pass

    def add_training_file(self):
        src = filedialog.askopenfile()
        dest = filedialog.askdirectory(initialdir=r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images')
        if not 1 != 1:
            shutil.copy(src.name, dest)
            pass
        else:
            messagebox.showinfo('Bu dosya mevcut')
        pass

    def add_test_image(self, parent, label_number):
        if label_number != None:
            with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images\Pilav',
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # the user changed their mind

                # save the current contents in the file
                pathname = fileDialog.GetPath()
                dest = r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images\test_images'
                if not os.path.exists(os.path.join(dest, os.path.split(pathname)[1])):
                    shutil.copy(pathname, dest)
                    self.cursor.execute('insert into tbl_01_01_testimage(labelnumber, foldername, filename) '
                                        'values(?, ?, ?)', label_number, dest, os.path.split(pathname)[1])
                    self.cnxn.commit()
                    pass
                else:
                    messagebox.showinfo('Bu dosya mevcut')
                    pass
        else:
            messagebox.showinfo('Lütfen sınıf seçiniz')

    def pre_process_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        return image

    def random_batch(self, batch_size, images_size, is_training=True, append_preprocess=False):
        images = []
        labels = []
        for i in range(batch_size):
            random_number = np.random.randint(0, images_size)
            if is_training:
                labels.append(self.training_labels[random_number])
                image = self.training_images[random_number]
            else:
                labels.append(self.test_labels[random_number])
                image = self.test_images[random_number]

            if append_preprocess:
                self.pre_process_image(image)

            images.append(image)
        return images, labels
        pass


class DataSinif:
    labelnumber= -1
    sinifname = ""
    foldername=""
    pass
