import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import SI2Conc_data
from src.original.MB_QBI_UoManchesterUK.QbiPy.dce_models.tissue_concentration import signal_to_concentration




# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MB_UoManchester_sig_to_conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):

    ##Prepare input data
    
    #The code eeds a baseline signal to calculate M0 from, so this will be the mean signal from 1 to BL points as used in the orginal data
    S0=np.mean(s_array[1:BLpts])

    #The code used to make the test data doesn't include the first point on the curve, so remove it here from s and conc and reduce BLpts by 1
    s_array=s_array[1:]
    conc_array=conc_array[1:]
    BLpts=BLpts-1

    #The code also needs a baseline signal to calculate M0 from, so this will be the mean signal from 1 to BL points

    #Relaxivity units should be per ms rather than per s for this code, so multiply r1 by 1000
    r1=r1*1000

    

    # run test

    conc_curve = signal_to_concentration(s_array, T1base, S0, fa, tr, r1, BLpts)[0]


    np.testing.assert_allclose( conc_curve, conc_array, rtol=r_tol, atol=a_tol )

