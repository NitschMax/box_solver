import help_functions as hf
import matplotlib.pyplot as plt
import numpy as np

import current_noise_ratio as cnr
import data_directory as dd
import setup as set


def main():
    np.set_printoptions(precision=3)

    t_set = set.create_transport_setup()

    #t_set.adjust_to_z_blockade()

    t_set.initialize_leads()
    t_set.initialize_box()
    t_set.connect_box()

    t_set.maj_box.print_eigenstates()

    sys = t_set.build_qmeq_sys()
    sys.solve(qdq=False, rotateq=False)

    points = 41

    current_condition = True
    t_set.blockade_condition_via_current = current_condition
    params = cnr.param_dict()
    params = cnr.fill_param_dict(params, t_set, points)

    recalculate = False

    t1 = hf.tunnel_from_gamma(t_set.gamma_01)
    x = np.linspace(-t_set.dphi, 2 * t1, points)
    y = x

    X, Y = np.meshgrid(x, y)

    I = dd.load_data(params, '0_blockade-tunnel-sweep')
    # Print all columns and rows of I independent of the size of the matrix
    # Put all values below 1e-4 to zero and all above to 1
    # Print each value together with its index
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    # How to print the index of each value next to it?
    # A: Use np.ndenumerate

    # idx (21, 22) is curropted, it seems like the optimization failed
    # redo the calculation for this point and save the array

    # For comparison, print values of I that are neighbors of idx (21, 22)
    print(I[20, 22])
    print(I[21, 21])
    print(I[21, 23])
    print(I[22, 22])

    new_val = cnr.sweep_func(t_set,
                             X[21, 22],
                             Y[21, 22],
                             0,
                             guess=np.array([0.5, 0, 0.5, 0]))

    I[21, 22] = new_val
    dd.save_data(I, params, '0_blockade-tunnel-sweep')


if __name__ == '__main__':
    main()
