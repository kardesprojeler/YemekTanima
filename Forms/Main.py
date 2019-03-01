import tkinter
from Models import Model as model
from Datas import Data as data
import wx
import Forms.frmSinifEkle as frm

class Main(wx.MDIParentFrame):
    def __init__(self):
        wx.MDIParentFrame.__init__(self, None, -1, "Resim Tanıma", size=(600, 400))
        self.data = data.Data()
        self.make_main_form()

    def make_main_form(self):
        menubar = wx.MenuBar()

        menu_data = wx.Menu()
        menu_data.Append(1000, "Sınıf Ekle")
        menu_data.Append(1001, "Eğitim Resmi Ekle")
        menu_data.Append(1002, "Test Resmi Ekle")
        menubar.Append(menu_data, "Veriler")

        self.Bind(wx.EVT_MENU, self.add_datasinif, id=1000)
        self.Bind(wx.EVT_MENU, self.data.add_training_file, id=1001)
        self.Bind(wx.EVT_MENU, self.data.add_test_file, id=1002)

        menu_model = wx.Menu()
        menu_model.Append(2000, "Modeli Oluştur")
        menu_model.Append(2001, "Modeli Eğit")
        menu_model.Append(2002, "Modeli Test Et")
        menubar.Append(menu_model, "Model")

        self.Bind(wx.EVT_MENU, self.make_model, id=2000)
        self.Bind(wx.EVT_MENU, self.train_model, id=2001)
        self.Bind(wx.EVT_MENU, self.test_model, id=2002)

        menu_model = wx.Menu()
        menu_model.Append(3000, "Hakkımızda")
        menu_model.Append(3001, "Yardım")

        self.SetMenuBar(menubar)

    def make_model(self, evt):
        #self.labelText.set('Model Oluşturuluyor...')
        self.model = model.Model()
        self.model.make_model()
        #self.labelText.set('Model Oluşturuldu')

    def train_model(self, evt):
        #self.status_label.config(text='Model Eğitiliyor...')
        #self.labelText.set('Model Eğitiliyor...')
        self.model.train_step(1000)
        #self.labelText.set('Eğitim Tamamlandı')

    def test_model(self, evt):
        self.model.test_accuracy()
        pass

    def add_datasinif(self, evt):
        form = frm.frmSinifEkle(self)
        form.Show(True)
    def add_file(self, evt):
        self.data.add_test_file()


if __name__ == '__main__':
    app = wx.App()
    frame = Main()
    frame.Show()
    app.MainLoop()