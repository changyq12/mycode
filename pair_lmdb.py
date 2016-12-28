import numpy as np
import lmdb
import caffe
import scipy.io as sio



# Let's pretend this is interesting data
#X = np.zeros((N, 6, 32, 32), dtype=np.uint8)
#y = np.zeros(N, dtype=np.uint8)
#X = numpy.fromfile("data.bin",dtype ='double')
#X = b.reshape(64,128,6,6051)
#X = b.T
data= sio.loadmat('datamat.mat')
X = data['Is']
y = data['y']
NTR=range(10000)
NTE=range(10001,13000)


# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
# N = length
map_size = X.nbytes * 10

env = lmdb.open('train_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in NTR:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

env = lmdb.open('test_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in NTE:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
