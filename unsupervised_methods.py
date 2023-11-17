import numpy as np
import scipy, math
from utils import detrend

def CHROM(RGB):
    Xcomp, Ycomp = (np.array([[3 ,  -2 , 0],
                            [1.5, 1,  -1.5]]) @ RGB.T)
    # print(Xcomp)
    # print(Ycomp)
    sX = np.std(Xcomp)
    sY = np.std(Ycomp)
    alpha = sX/sY
    alpha = np.repeat(alpha, Xcomp.shape[0], 0)
    bvp = Xcomp - np.multiply(alpha, Ycomp)
    return bvp

def GRAYSCALE(RGB):
    signal = (np.array([[0.299, 0.587, 0.114]]) @ RGB.T)
    # print(signal.shape)
    return signal

def Green(RGB):
    # print(RGB)
    # print(RGB[:][1])
    signal = (np.array([[0, 1, 0]]) @ RGB.T)
    # print(signal.shape)
    return signal

def POS(RGB, fs=30):
    WinSec = 1.6
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    return BVP

def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = scipy.linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S

def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat

def ICA(RGB, FS=30):
    RGB = (RGB-RGB.mean())/RGB.std(axis=0)
    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    for c in range(3):
        BGRDetrend = detrend(RGB[:, c])
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    return np.array(BVP_I).real[0]

def lgi(RGB):
        pulse = []
        frames = len(RGB)
        C = []
        # print("RGB: ",RGB[0])
        # a=RGB[0]

        for f in range(frames):
            x = RGB[f]
            # print(x.shape)
            C.append([np.mean(x[:][0]), np.mean(x[:][1]), np.mean(x[:][2])])

        C_=np.transpose(C).astype(np.float32)
        U, E, V = np.linalg.svd(C_)
        S = U[:,0]
        P = np.eye(3) - np.outer(S, S)
        F = np.zeros((frames, 3))
        for f in range(frames):
            # print(C)
            # print(C[f][:])
            # print(F[f,:])
            F[f,:] = np.dot(P, np.transpose(C[f][:]))
        pulse = list(map(float, F[:,1]))
        return pulse

def pbv(RGB):
    # print(RGB.shape)
    std_rgb=np.std(np.array([[1, 1, 1]]) @ RGB.T)
    std_red=np.std(np.array([[1, 0, 0]]) @ RGB.T)
    std_green=np.std(np.array([[0, 1, 0]]) @ RGB.T)
    std_blue=np.std(np.array([[0, 0, 1]]) @ RGB.T)

    var_red=np.var(np.array([[1, 0, 0]]) @ RGB.T)
    var_green=np.std(np.array([[0, 1, 0]]) @ RGB.T)
    var_blue=np.std(np.array([[0, 0, 1]]) @ RGB.T)

    pbv_t = std_rgb/(var_red+var_green+var_blue)
    pbv_t_=var_red+var_green+var_blue
    # print(pbv_t)
    # print(pbv_t_)
    # pbv_t = pbv_t.reshape(-1, 1)
    pbv_t=np.eye(3)*pbv_t

    X = np.vstack((np.array([[1, 0, 0]]) @ RGB.T, np.array([[0, 1, 0]]) @ RGB.T, np.array([[0, 0, 1]]) @ RGB.T))
    P = np.diag([std_red, std_green, std_blue])
    # M = np.dot(np.linalg.inv(np.dot(P, X)), P)


    Pbv_XXT = np.dot(P @ pbv_t, X @ X.T)
    M = np.dot(np.linalg.inv(Pbv_XXT), P)

    k = np.linalg.norm(P @ pbv_t)
    signal = k * np.dot(M, RGB.T)
    return signal
