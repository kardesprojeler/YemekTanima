import wx

class MDIFrame(wx.MDIParentFrame):

    def __init__(self):
        wx.MDIParentFrame.__init__(self, None, -1, "MDI Parent",size=(600,400))
        menu = wx.Menu()
        menu.Append(5000, "&New Window 1")
        menu.Append(5002, "&New Window 2")
        menu.Append(5001, "E&xit")
        menubar = wx.MenuBar()
        menubar.Append(menu, "&File")
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnNewWindow1, id=5000)
        self.Bind(wx.EVT_MENU, self.OnExit, id=5001)
        self.Bind(wx.EVT_MENU, self.OnNewWindow2, id=5002)

    def OnExit(self, evt):
        self.Close(True)

    def OnNewWindow1(self, evt):
        #win = wx.MDIChildFrame(self, -1, "Child Window")
        win = MDIHijo1(self)
        win.Show(True)

    def OnNewWindow2(self, evt):
        #win = wx.MDIChildFrame(self, -1, "Child Window")
        win = MDIHijo2(self)
        win.Show(True)


class MDIHijo1(wx.MDIChildFrame):

    def __init__(self,parent):
        wx.MDIChildFrame.__init__(self,parent,title='Ventana Hijo 1')
        menu = parent.GetWindowMenu()
        menu.Append(5500, "&Son 1")
        menu.Append(5501, "&Son 1")
        # menubar = wx.MenuBar()
        # menubar.Append(menu, "&File")
        # self.SetMenuBar(menubar)


class MDIHijo2(wx.MDIChildFrame):

    def __init__(self,parent):
        wx.MDIChildFrame.__init__(self,parent,title='Ventana Hijo 2')
        menu = parent.GetWindowMenu()
        menu.Append(5500, "&Son 2")
        menu.Append(5501, "&Son 2")
        # menubar = wx.MenuBar()
        # menubar.Append(menu, "&File")
        # self.SetMenuBar(menubar)


app = wx.PySimpleApp()
frame = MDIFrame()
frame.Show()
app.MainLoop()