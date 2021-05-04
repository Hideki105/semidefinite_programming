import numpy   as np
import scipy.linalg as linalg
import os
import time

from util import readcsv, savecsv, savefig

class sdp_interior_point():
    def __init__(self,A,b,C):
        #初期化処理
        m = len(A)
        n = C.shape[0]
        X0 = np.eye(n)
        y0 = np.zeros(m)
        Z0 = np.eye(n)

        self.X0 = X0
        self.y0 = y0
        self.Z0 = Z0
        self.b = b
        self.C = C
        self.Q = self.set_vec2svec_matrix(n)

        self.m = m
        self.n = n
        
        
        self.matA = np.zeros((m,n**2))
        for i in range(len(A)):
            vecAiT = self.vectorize(A[i]).transpose()
            self.matA[i,:] = vecAiT

    def set_vec2svec_matrix(self,n):
        #対称クロネッカー積における変換行列Q
        """
        行列Aが実対称行列であるとき、変換行列Qは下記の性質をもつ。

        svec(A) = Q    vec(A)
        vec(A)  = Q^T svec(A)
        """
        Q = np.zeros((int(n*(n+1)/2) , int(n**2)))

        for k in range(n):
            for l in range(n):
                if k == l:
                    i = int(n*l -l*(l+1)/2 + k)
                    j = int(n*l + k)
                    Q[i,j] = 1
                elif k > l:
                    i = int(n*l -l*(l+1)/2 + k )
                    j = int(n*l + k)
                    Q[i,j] = 1/np.sqrt(2)
                elif k < l:
                    i = int(n*k -k*(k+1)/2 + l)
                    j = int(n*l + k)
                    Q[i,j] = 1/np.sqrt(2) 
        return Q

    def vectorize(self,A):
        #vec作用素
        """
        vec作用素を用いて行列Aのベクトルを返す。
        A = |a11 a12|
            |a21 a22|
        vec(A) = [a11,a21,a12,a22]^T
        """
        vecA = A.transpose().flatten()
        return vecA

    def skronecker(self,A,Q):
        #対称クロネッカー積
        """
        行列Aと変換行列Qを与えられたとき対称クロネッカー積を返す。
        X*I = 1/2*Q (IxA + AxI) Q^T
        """
        n = self.n
        I = np.eye(n) 
        return 1/2*np.dot(np.dot(Q, np.kron(I,A) + np.kron(A,I)), Q.transpose())

    def solve(self):
        #主双対内点法による数値解析
        return

    def mu(self,Xk,Zk):
        return np.trace(np.dot(Xk,Zk.transpose())) / Xk.shape[0]
    
    def w(self,Xk,yk,b,C):
        #双対ギャップの算出
        return np.trace(np.dot(C,Xk.transpose())) - np.dot(b,yk)
    
    def log(self,Xk,yk,Zk,muk,wk,dt,logpath="."):
        #主双対内点法による処理を記録
        """
        mu_ | 主変数xと双対変数zの平均値xT*z/N        
        w_  | 双対ギャップcT*x - bT*y
        cx_ | 主問題の目的関数cT*x
        by_ | 双対問題の目的関数bT*y
        dt  | 1反復分の計算時間
        x_  | k反復目の主変数xk
        y_  | k反復目の主変数yk
        z_  | k反復目の主変数zk
        """

        mu_= muk.reshape(1,muk.size)[0]
        w_ = wk.reshape(1,wk.size)[0]
        cx_= np.trace(np.dot(self.C,Xk.transpose()))
        by_= np.dot(self.b,yk)
        x_ = Xk.reshape(1,Xk.size)[0]
        y_ = yk.reshape(1,yk.size)[0]
        z_ = Zk.reshape(1,Zk.size)[0]
        print(mu_,w_,cx_,by_,dt,x_,y_,z_)
        d_ = np.hstack([mu_,w_,cx_,by_,dt,x_,y_,z_])
        savecsv(d_,logpath)
        return


class sdp_aho_xzpzx(sdp_interior_point):
    """
    日付: 2021/05/04
    概要:
    半正定値計画の主双対内点法の主双対パス追跡法
    AHO方向のXZ+ZX method
    
    min  trace(C,X)
    s.t. trace(Ai,X) = bi, Xi>=0
    　　　
    主双対内点法による処理を記録する。
    
    mu_ | 主変数xと双対変数zの平均値xT*z/N        
    w_  | 双対ギャップcT*x - bT*y
    cx_ | 主問題の目的関数cT*x
    by_ | 双対問題の目的関数bT*y
    dt  | 1反復分の計算時間
    x_  | k反復目の主変数xk
    y_  | k反復目の主変数yk
    z_  | k反復目の主変数zk

    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","sdp_aho_xzpzx")
    #半正定値計画
    problem = sdp_aho_xzpzx(A,b,C)
    #主双対内点法による数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(path=LOGPATH)
    """

    def sdp_path(self,matA,b,C,Xk,yk,Zk,sigm_k=0.01,tau=0.95):
        #主双対内点法、主双対パス追跡法
        m   = self.m
        n   = self.n
        Q   = self.Q 
        muk = self.mu(Xk,Zk)
        
        vecXk = self.vectorize(Xk)
        vecZk = self.vectorize(Zk)
        vecC  = self.vectorize(C)

        rp = b - np.dot(matA,vecXk)
        rd = - np.dot(matA.transpose(),yk) - vecZk + vecC
        Rc = 0.9*muk*np.eye(n) - 1/2 *(np.dot(Xk,Zk)+np.dot(Zk,Xk))
        rc = self.vectorize(Rc)

        rd_ = np.dot(Q,rd)
        rc_ = np.dot(Q,rc)
        A_  = np.dot(matA,Q.transpose())
        
        #　探索方向の算出
        E = self.skronecker(Zk,Q)
        F = self.skronecker(Xk,Q)
        E_inv = np.linalg.inv(E)
        
        M     = np.dot(A_,np.dot(E_inv,np.dot(F,A_.transpose())))
        LU = linalg.lu_factor(M)
        r_  = rp + np.dot(np.dot(A_,E_inv), np.dot(F,rd_) - rc_ ) 
        dy = linalg.lu_solve(LU, r_)
        
        svecdZ  = rd_ - np.dot(A_.transpose(),dy)
        svecdX  = - np.dot(E_inv, np.dot(F,rd_ - np.dot(A_.transpose(),dy) ) - rc_)
        
        # svec(X)をvec(X)に変換
        vecdX = np.dot(Q.transpose(),svecdX)
        vecdZ = np.dot(Q.transpose(),svecdZ)

        # vec(X)をXに変換
        dX = vecdX.reshape(n,n)
        dZ = vecdZ.reshape(n,n)

        #　ステップ幅の算出
        def calc_step(Xk,dX):
            Lx = np.linalg.cholesky(Xk)
            Lx_inv = np.linalg.inv(Lx)
            LxT_inv= np.linalg.inv(Lx.transpose())

            X = np.dot(Lx_inv,np.dot(dX,LxT_inv))
            lambdas, _ = np.linalg.eig(X)

            if np.min(lambdas) < 0:
                return np.min([0.9, -0.9/np.min(lambdas)])
            else:
                return 0.9

        alpha_p = calc_step(Xk,dX)
        alpha_d = calc_step(Zk,dZ)
        
        # 解の更新
        Xk = Xk + alpha_p*dX
        yk = yk + alpha_d*dy
        Zk = Zk + alpha_d*dZ
        return Xk, yk, Zk
    
    
    def solve(self,vervose=False,logpath=".",eps=1e-6):
        #主双対内点法による数値解析
        t0 = time.time()
        matA  = self.matA
        b  = self.b
        C  = self.C

        Xk = self.X0
        yk = self.y0
        Zk = self.Z0


        muk= self.mu(Xk,Zk)
        wk = self.w(Xk,yk,b,C)
        
        t1 = time.time()
        dt = t1 -t0

        if vervose == True:
            csvpath = os.path.join(logpath,"log.csv")
            if os.path.isfile(csvpath):
                os.remove(csvpath)
            self.log(Xk,yk,Zk,muk,wk,dt,logpath)

        for i in range(200):
            t0 = time.time()
            Xk,yk,Zk = self.sdp_path(matA,b,C,Xk,yk,Zk)
            t1 = time.time()
            dt = t1 -t0
            muk= self.mu(Xk,Zk)
            wk = self.w(Xk,yk,b,C)
            
            if vervose == True:
                self.log(Xk,yk,Zk,muk,wk,dt,logpath)
            elif np.abs(muk) < eps:
                break
        return Xk,yk,Zk

if __name__ == "__main__":
    
    C = np.array([[1,2,3],[2,9,0],[3,0,7]])
    
    A = []
    A.append(np.array([[1, 0, 1],[0, 3, 7],[1, 7, 5]]))
    A.append(np.array([[0, 2, 8],[2, 6, 0],[8, 0, 4]]))
    
    b = []
    b.append(11)
    b.append(19)
    
    #主双対パス追跡法_AHO_XZ+ZAmethod(通称AHO方向)
    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","sdp_aho_xzpzx")
    if not os.path.isdir(LOGPATH):
        os.mkdir(LOGPATH)
    #半正定値計画
    problem = sdp_aho_xzpzx(A,b,C)
    #数値解析
    Xk,yk,Zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    savefig(logpath=LOGPATH)