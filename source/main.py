from source.qbca import QBCA
from source.utils import read_datafile

file = 'test.csv'
data = read_datafile(file)

if __name__ == '__main__':
    QBCA()._quantization(data)
