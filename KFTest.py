from KalmanFilter.KFFunction import KFTesting
import hdf5storage as h

test = h.loadmat('data/GETestData18.mat')

KFData =  test['LSandKFTestData'][2]

KFCoeffs = [1.95, -0.95]

KFMSEE, KFMSEP = KFTesting(KFData, KFCoeffs)
print(KFMSEP)