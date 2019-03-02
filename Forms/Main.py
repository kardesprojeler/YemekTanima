import tkinter
from Models import Model as model
from Datas import Data as data
import wx
import Forms.frmTestImage as frmImage
import Forms.frmSinifEkle as frmSinif


class Main(wx.MDIParentFrame):
    def __init__(self):
        wx.MDIParentFrame.__init__(self, None, -1, "Resim Tanıma", size=(600, 400))
        self.data = data.Data()
        self.model = model.Model()
        self.make_main_form()

    def make_main_form(self):
        menubar = wx.MenuBar()

        menu_data = wx.Menu()
        menu_data.Append(1000, "Sınıf Ekle")
        menu_data.Append(1001, "Eğitim Resmi Ekle")
        menu_data.Append(1002, "Test Resmi Ekle")
        menubar.Append(menu_data, "Veriler")

        self.Bind(wx.EVT_MENU, self.add_data_sinif, id=1000)
        self.Bind(wx.EVT_MENU, self.data.add_training_file, id=1001)
        self.Bind(wx.EVT_MENU, self.add_test_file, id=1002)

        menu_model = wx.Menu()
        menu_model.Append(2000, "Modeli Oluştur")
        menu_model.Append(2001, "Modeli Eğit")
        menu_model.Append(2002, "Modeli Test Et")
        menu_model.Append(2003, "Tahmin Yap")

        menubar.Append(menu_model, "Model")

        self.Bind(wx.EVT_MENU, self.make_model, id=2000)
        self.Bind(wx.EVT_MENU, self.train_model, id=2001)
        self.Bind(wx.EVT_MENU, self.test_model, id=2002)
        self.Bind(wx.EVT_MENU, self.test_model_for_one_image, id=2003)

        menu_model = wx.Menu()
        menu_model.Append(3000, "Hakkımızda")
        menu_model.Append(3001, "Yardım")

        self.SetMenuBar(menubar)

    def make_model(self, evt):
        #self.labelText.set('Model Oluşturuluyor...')
        self.model.make_model()
        wx.MessageBox('Model Oluşturuldu', 'Bilgilendirme', wx.OK | wx.ICON_INFORMATION)

    def train_model(self, evt):
        #self.status_label.config(text='Model Eğitiliyor...')
        #self.labelText.set('Model Eğitiliyor...')
        self.model.train_step(400)
        #self.labelText.set('Eğitim Tamamlandı')

    def test_model(self, evt):
        self.model.test_accuracy()
        pass

    def add_data_sinif(self, evt):
        form = frmSinif.frmSinifEkle(self)
        form.Show(True)

    def add_test_file(self, evt):
        form = frmImage.frmTestImage(self)
        form.Show(True)

    def test_model_for_one_image(self, evt):
        self.model.test_accuracy_for_one_image()


if __name__ == '__main__':
    app = wx.App()
    frame = Main()
    frame.Show()
    app.MainLoop()