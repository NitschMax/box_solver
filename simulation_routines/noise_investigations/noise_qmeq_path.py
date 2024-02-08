# import package via sys.path
import os
import sys


def noise_qmeq_path():
    home_dir = os.path.expanduser("~")
    sys.path.append(home_dir +
                    'Documents/projects/majorana_box/noise_calculations/qmeq')
