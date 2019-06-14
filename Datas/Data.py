import os
import shutil
from pyodbc import connect
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wx
from PIL import Image as pilimage
from absl import flags
from enum import Enum
FLAGS = flags.FLAGS
import cv2
import Datas.SelectiveSearch as selectivesearch

fields = 'Sınıf İsmi', 'Klasör İsmi'
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=LAPTOP-1CAUHSG4;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=True;'
    )

cnxn = connect(conn_str)
cursor = cnxn.cursor()


def get_sinif_list():
    array = []
    cursor.execute("select * from tbl_01_01_sinif")
    row = cursor.fetchone()
    while row:
        sinif = DataSinif()
        sinif.id = row[0]
        sinif.sinifname = row[1]
        sinif.foldername = row[2]
        sinif.fiyat = row[3]
        array.append(sinif)
        row = cursor.fetchone()
    return array


def one_hot_label(sinif, siniflist):
    label = np.zeros((siniflist.__len__(),), dtype=int)
    finding_index = siniflist.index(sinif)
    label[finding_index] = 1
    return label
    pass

def read_train_images(heigh, width, batch_size):
    siniflist = get_sinif_list()
    images = []
    labels = []
    for sinif in siniflist:
        path = os.path.join(os.getcwd(), "images",  sinif.foldername)
        for filename in os.listdir(path):
            file_content = cv2.imread(os.path.join(path, filename))
            if file_content is not None:
                im = cv2.resize(file_content, dsize=(heigh, width), interpolation=cv2.INTER_CUBIC)
                images.append(im)
                labels.append(siniflist.index(sinif))
            pass
        pass
    images = np.array(images)
    images = images / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(len(images)).batch(batch_size=batch_size)

    """
    train_dataset = []
    batch = []
    i = 0
    for data in dataset:
        i = i + 1
        if batch_size == i:
            train_dataset.append(batch)
            batch = []
            i = 0
            pass
        batch.append(data)
    """
    return train_ds


def read_test_images(heigh, width, batch_size):
    siniflist = get_sinif_list()
    images = []
    labels = []
    for sinif in siniflist:
        cursor.execute('select id, foldername, filename from tbl_01_01_testimage a where id = ?', sinif.id)
        row = cursor.fetchone()
        while row:
            path = os.path.join(os.getcwd(), "images", row[1], row[2])
            file_content = cv2.imread(path)
            if file_content is not None:
                im = cv2.resize(file_content, dsize=(heigh, width), interpolation=cv2.INTER_CUBIC)
                images.append(im)
                labels.append(siniflist.index(sinif))
            row = cursor.fetchone()
        pass
    images = np.array(images)
    images = images / 255.0
    buffer_size = 5
    if len(images) > 0:
        buffer_size = len(images)
        pass
    test_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(buffer_size).batch(batch_size=batch_size)
    """
    test_dataset = []
    batch = []
    i = 0
    for data in dataset:
        i = i + 1
        if batch_size == i:
            test_dataset.append(batch)
            batch = []
            i = 0
            pass
        batch.append(data)
        """
    return test_ds


def insertsinif(sinifname, foldername):
    cursor.execute("select * from tbl_01_01_sinif where (sinifname = ? or " +
                   "foldername = ?)", sinifname, foldername)
    row = cursor.fetchone()
    if row:
        wx.MessageBox('Böyle Bir Satır Mevcut', 'Attention', wx.OK | wx.ICON_WARNING)
        pass
    else:
        cursor.execute("insert into tbl_01_01_sinif(sinifname, foldername) " +
                       "values(?, ?)", sinifname, foldername)
        cnxn.commit()
    pass

def getsinifcount():
    cursor.execute("select sinifcount = count(*) from tbl_01_01_sinif")
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    return 0
    pass

def add_data_sinif(sinifname, foldername):
    if sinifname is not '' and foldername is not '':
        if not os.path.exists(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername):
            os.mkdir(r"C:/Users/BULUT/Documents/GitHub/YemekTanima/images/" + foldername)
            insertsinif(foldername, sinifname)
            pass
        else:
            wx.MessageBox('Bu sınıf ismi mevcut', 'Attention', wx.OK | wx.ICON_WARNING)
            pass
        pass
    else:
        wx.MessageBox('Alanları doldurunuz!', 'Attention', wx.OK | wx.ICON_WARNING)
        pass
    pass

def add_training_file():
    src = filedialog.askopenfile()
    dest = filedialog.askdirectory(initialdir=r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images')
    if not os.path.exists(os.path.join(dest, os.path.split(src)[1])):
        shutil.copy(src.name, dest)
        pass
    else:
        wx.MessageBox('Bu dosya mevcut!', 'Attention', wx.OK | wx.ICON_WARNING)
    pass

def add_test_image(parent, label_number):
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
                cursor.execute('insert into tbl_01_01_testimage(labelnumber, foldername, filename) '
                                    'values(?, ?, ?)', label_number, dest, os.path.split(pathname)[1])
                cnxn.commit()
                pass
            else:
                messagebox.showinfo('Bu dosya mevcut')
                pass
    else:
        messagebox.showinfo('Lütfen sınıf seçiniz')


def pre_process_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    return image


def read_image(width, height):
    with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Documents\GitHub\YemekTanima\images',
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind

        # save the current contents in the file
        pathname = fileDialog.GetPath()

        file_content = pilimage.open(pathname)
        im = file_content.resize((width, height), pilimage.ANTIALIAS)
        im = [np.array(im)]
        im = np.array(im).reshape((-1, width, height, 3))

        return im

def get_fragment_tray_images(height, width):
    with wx.FileDialog(None, 'Open', r'C:\Users\BULUT\Desktop',
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return  # the user changed their mind

        return fragment_tray_image(fileDialog.GetPath(), height, width)


def fragment_tray_image(path, height, width):
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

    delete_supererogator_rect(candidates)
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


def delete_supererogator_rect(candidates):
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


def random_batch(batch_size, images, labels, append_preprocess=False):
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        random_number = np.random.randint(0, images.__len__())
        batch_labels.append(labels[random_number])
        img = images[random_number]

        if append_preprocess:
            pre_process_image(img)

        batch_images.append(img)
    return batch_images, batch_labels


class DataSinif:
    id = -1
    sinifname = ""
    foldername = ""
    fiyat = 0
    pass


class SimpleModelFlags(Enum):
    buffer_size = 1000
    batch_size = 10
    init_filter = (3, 3)
    stride = (1, 1)
    save_path = None
    mode = 'from_depth'
    depth_of_model = 7
    growth_rate = 12
    num_of_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    data_format = 'channels_last'
    bottleneck = True
    compression = 0.5
    weight_decay = 1e-4
    dropout_rate = 0.
    pool_initial = False
    include_top = True
    train_mode = 'custom_loop'
    image_height = 120
    image_width = 120
    image_deep = 3


class DenseNetFlags(Enum):
    buffer_size = 1000
    batch_size = 10
    init_filter = (3, 3)
    stride = (1, 1)
    save_path = None
    mode = 'from_depth'
    depth_of_model = 7
    growth_rate = 12
    num_of_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    data_format = 'channels_last'
    bottleneck = True
    compression = 0.5
    weight_decay = 1e-4
    dropout_rate = 0.
    pool_initial = False
    include_top = True
    train_mode = 'custom_loop'
    image_height = 120
    image_width = 120
    image_deep = 3

class GeneralFlags(Enum):
    epoch = 1
    enable_function = False,
    train_mode = 'custom_loop'
    checkpoint_dir = os.path.join('checkpoints', 'yemek_tanima')