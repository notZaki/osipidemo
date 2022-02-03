import pytest
import numpy as np

from ..helpers import osipi_parametrize, log_init, log_results
from . import t1_data
from osipi_code_collection.original.McGill_Can.vfa import despot, novifast


# All tests will use the same arguments and same data...
arg_names = 'label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol'
test_data = (
    t1_data.t1_brain_data() +
    t1_data.t1_quiba_data() +
    t1_data.t1_prostate_data()
    )

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    filename_prefix = 'TestResults_T1mapping'
    log_init(filename_prefix, '_test_mcgill_t1_novifast', ['label', 'r1_ref', 'r1_measured'])
    log_init(filename_prefix, '_test_mcgill_t1_VFA_lin', ['label', 'r1_ref', 'r1_measured'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_mcgill_t1_novifast(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:
        
    # prepare input data
    tr = tr_array[0]
    fa_array_rad = fa_array * np.pi/180.
    
    # run test (non-linear)
    [s0_nonlin_meas, t1_nonlin_meas] = novifast(s_array,fa_array_rad,tr)
    r1_nonlin_meas = 1./t1_nonlin_meas[0]  
    np.testing.assert_allclose( [r1_nonlin_meas], [r1_ref], rtol=r_tol, atol=a_tol )
    log_results(filename_prefix, '_test_mcgill_t1_novifast', [label, r1_ref, r1_nonlin_meas])


# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['Pat5_voxel5_prostaat'])
def test_mcgill_t1_VFA_lin(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:
    #   Expected fails: 1 low-SNR prostate voxel
    
    # prepare input data
    tr = tr_array[0]
    fa_array_rad = fa_array * np.pi/180.
    
    # run test (non-linear)
    [s0_lin_meas, t1_lin_meas] = despot(s_array,fa_array_rad,tr)
    r1_lin_meas = 1./t1_lin_meas    
    np.testing.assert_allclose( [r1_lin_meas], [r1_ref], rtol=r_tol, atol=a_tol )
    log_results(filename_prefix, '_test_mcgill_t1_VFA_lin', [label, r1_ref, r1_lin_meas])
