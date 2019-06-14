import numpy as np


def main():
    print('1H', radst_to_hzg(267.522e6))
    print('13C', radst_to_hzg(67.283e6))
    print('14N', radst_to_hzg(19.338e6))
    print('15N', radst_to_hzg(-27.126e6))


def radst_to_hzg(gamma):
    gamma = gamma * 1e-4 / 2 / np.pi
    return '{:.3E}'.format(gamma)


if __name__ == '__main__':
    main()
