<<<<<<< HEAD
from pyodbc import connect
from tkinter import messagebox
=======
import pyodbc as odbc
from tkinter import messagebox, filedialog, ttk
import shutil
import os
import numpy as np
from PIL import Image as pilimage
import tensorflow as tf
import wx
import Datas.SelectiveSearch as selectivesearch
import matplotlib.pyplot as plt
>>>>>>> 3a007f1799aae5a4e2cbeca31a6c08a20ea39dd9

fields = 'Sınıf İsmi', 'Klasör İsmi'
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=localhost\SQLEXPRESS;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=yes;'
    )

class Data:
    cnxn = connect(conn_str)
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
            sinif.id = row[0]
            sinif.sinifname = row[1]
            sinif.foldername = row[2]
            sinif.fiyat = row[3]
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
            self.cursor.execute('select id, foldername, filename from tbl_01_01_'
                                'testimage a where id = ?', sinif.id)
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
<<<<<<< HEAD
        pa
        ss
    pass
=======
        pass
    def getsinifcount(self):
        self.cursor.execute("select sinifcount = count(*) from tbl_01_01_sinif")
        row = self.cursor.fetchone()
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
        if not os.path.exists(os.path.join(dest, os.path.split(src)[1])):
            shutil.copy(src.name, dest)
            pass
        else:
            wx.MessageBox('Bu dosya mevcut!', 'Attention', wx.OK | wx.ICON_WARNING)
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

    def read_image(self, width, height):
        with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()

            file_content = pilimage.open(pathname)
            im = file_content.resize((width, height), pilimage.ANTIALIAS)
            im = [np.array(im)]
            im = np.array(im).reshape(1, width, height, 3)

            return im

    def get_fragment_tray_images(self, height, width):
        with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Desktop',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            return self.fragment_tray_image(fileDialog.GetPath(), height, width)

    def fragment_tray_image(self, path, height, width):
        img = pilimage.open(path)
        img = img.resize((250, 250), pilimage.ANTIALIAS)
        im = np.asarray(img, dtype='uint8')
        img_lbl, regions = selectivesearch.selective_search(
            im, scale=1, sigma=0.8, min_size=50)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 1000:
                continue
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])

        self.delete_supererogator_rect(candidates)
        fragmented_images = []
        fig = plt.figure(figsize=(8, 8))
        i = 0
        for x, y, w, h in candidates:
            im = img.crop((x, y, (x + w), (y + h)))
            i = i + 1
            fig.add_subplot(6, 6, i)
            plt.imshow(im)
            im.save(os.path.join(r"C:\Users\BULUT\Desktop", "indir(1).jpg"))
            im = im.resize((width, height), pilimage.ANTIALIAS)
            im = [np.array(im)]
            im = np.array(im).reshape((-1, width, height, 3))
            fragmented_images.append(im)
        plt.show()
        return fragmented_images

    def delete_supererogator_rect(self, candidates):
        deleted = set()
        for rectx in candidates:
            x, y, w, h = rectx
            for rect in candidates:
                if rect != rectx:
                    x0, y0, w0, h0 = rect
                    if x0 >= x and (x0 + w0) <= (x + w) and y0 >= y and (y0 + h0) <= (y + h):
                        if rectx not in deleted:
                            deleted.add(rectx)

        for i in deleted:
            candidates.remove(i)
>>>>>>> 3a007f1799aae5a4e2cbeca31a6c08a20ea39dd9
class DataSinif:
    id = -1
    sinifname = ""
    foldername = ""
    fiyat = 0
    pass
