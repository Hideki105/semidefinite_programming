import matplotlib.pyplot as plt
import numpy   as np
import os

def readcsv(csvfn):
    #csvファイルの読み出し
    d_ = np.loadtxt(csvfn,delimiter=",")
    return d_

def savecsv(d_,logpath="."):
    #csvファイルの書き出し
    logpath = os.path.join(logpath,"log.csv")
    with open(logpath,"a") as f:
        np.savetxt(f,d_.reshape(1,d_.size), delimiter=",")
    return

def savefig(logpath="."):
    #記録した主双対内点法のグラフ化
    csvpath = os.path.join(logpath,"log.csv")
    d_ = readcsv(csvpath)
    for i in range(d_.shape[1]):
        plt.figure()
        plt.plot(d_[:,i],"-o")
        plt.grid(True)
        plt.xlabel("iteration")
        pngpath = os.path.join(logpath,"row{0}.png".format(i))
        plt.savefig(pngpath)
        plt.close()
