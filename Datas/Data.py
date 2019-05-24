from pyodbc import connect
from tkinter import messagebox

conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=localhost\SQLEXPRESS;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=yes;'
    )

class Data:
    cnxn = connect(conn_str)
    cursor = cnxn.cursor()
    datasiniflist = []

    @classmethod
    def ReadSinif(x):
        Data.cursor.execute("select * from blt_01_01_sinif")
        row = Data.cursor.fetchone()
        while row:
            dtSinif = DataSinif()
            dtSinif.ref = row[0]
            dtSinif.labelnumber = row[1]
            dtSinif.sinifname = row[2]
            dtSinif.foldername = row[3]
            Data.datasiniflist.append(dtSinif)
            row = Data.cursor.fetchone()
            pass
        return Data.datasiniflist
        pass
    @classmethod
    def InsertSinif(cls, labelnumber, sinifname, foldername):
        Data.cursor.execute("select * from blt_01_01_sinif where (sinifname = ? or " +
                            "foldername = ? or labelnumber = ?)", sinifname, foldername, labelnumber)
        row = Data.cursor.fetchone();
        if row:
            messagebox.showinfo("Uyarı!", "Böyle Bir Satır Mevcut")
            pass
        else:
            Data.cursor.execute("insert into blt_01_01_sinif(ref, labelnumber, sinifname, foldername) " +
                                "values(?, ?, ?, ?)", 3, labelnumber, sinifname, foldername)
            Data.cnxn.commit()
            pass
        pa
        ss
    pass
class DataSinif:
    ref = -1
    labelnumber= -1
    sinifname = ""
    foldername=""
    pass
