from scipy.constants import c, h, electron_volt


def ev_to_nm(ev):
    lambd = h * c / ev / electron_volt
    print('{:.2f} eV = {:.2f} nm'.format(ev, lambd * 1e9))


def main():
    ev_to_nm(5.5)


if __name__ == '__main__':
    main()
