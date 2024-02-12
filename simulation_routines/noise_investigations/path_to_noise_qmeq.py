# import package via sys.path
import os
import sys


def noise_qmeq_path():
    home_dir = os.path.expanduser("~")
    ## Append path to begining not end of python path
    #sys.path.insert(
    #    1,
    #    home_dir + '/Documents/projects/majorana_box/noise_calculations/qmeq')
    #print(sys.path)
