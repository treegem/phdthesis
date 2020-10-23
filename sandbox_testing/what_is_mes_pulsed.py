import numpy as np
import scipy.io as sio


def main():
    data = sio.loadmat('mes_pulsed.mat')
    for key in data.keys():
        try:
            print(key, data[key])
        except AttributeError:
            pass
    print(np.unique(data['ws']))
    print(data['nvlist'])


if __name__ == '__main__':
    main()
