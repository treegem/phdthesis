def calc_polarization(n):
    polarization = 1 - 0.7 ** n
    if type(polarization) in [float, int]:
        print('polarization: {:.4}'.format(polarization))
    return polarization


def percentage_of_polarized_states(polarization):
    perc_of_polrzd_states = (1. + polarization) / 2.
    if type(perc_of_polrzd_states) in [float, int]:
        print('percentage of polarized states: {:.4}'.format(perc_of_polrzd_states))
    return perc_of_polrzd_states


def main():
    n_cycles = 5

    polarization = calc_polarization(n_cycles)

    percentage_of_polarized_states(polarization)


if __name__ == '__main__':
    main()
