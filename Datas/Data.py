import pyodbc as odbc
from tkinter import messagebox

conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=localhost\SQLEXPRESS;'
    r'DATABASE=YemekTanima;'
    r'Trusted_Connection=yes;'
    )

class Data:
    cnxn = odbc.connect(conn_str)
    cursor = cnxn.cursor()
    datasiniflist = []

    @classmethod
    def ReadSinif(x):
        Data.cursor.execute("select * from tbl_01_01_sinif")
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
    def InsertSinif(cls, sinifname, foldername):
        Data.cursor.execute("select * from tbl_01_01_sinif where (sinifname = ? or " +
                            "foldername = ?)", sinifname, foldername)
        row = Data.cursor.fetchone();
        if row:
            messagebox.showinfo("Uyarı!", "Böyle Bir Satır Mevcut")
            pass
        else:
            Data.cursor.execute("insert into tbl_01_01_sinif(sinifname, foldername) " +
                                "values(?, ?)", sinifname, foldername)
            Data.cnxn.commit()
            pass
        pass
    pass
class DataSinif:
    ref = -1
    labelnumber= -1
    sinifname = ""
    foldername=""
    pass
