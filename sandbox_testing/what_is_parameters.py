import scipy.io as sio


def main():
    data = sio.loadmat('parameters.mat')
    for key in data.keys():
        try:
            print(key, data[key])
        except AttributeError:
            pass


if __name__ == '__main__':
    main()
