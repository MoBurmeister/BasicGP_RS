import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import bo_optimizer as bo_opt
import model_setup as mo_set

st. set_page_config(layout="wide")

col_t1, col_t2, col_t3 = st.columns([0.5,15,0.5])

with col_t2:
    st.title('MVP AI at RuhlaSmart- Interface Bayesian Optimization with Gaussian Processes')

for _ in range(3):
    st.text("")

# Ensure the initial state is set up in session_state
if 'model_plot' not in st.session_state or 'acq_plot' not in st.session_state:
    st.session_state['model_plot'] = None
    st.session_state['acq_plot'] = None
    st.session_state['gp_model'] = None
    st.session_state['bounds'] = None
    st.session_state['train_X'] = None
    st.session_state['train_Y'] = None
    st.session_state['acq_function'] = None



#necessary initialization function:
def initialization_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_Y, bounds = mo_set.setup_datasets(dtype= torch.float64, device=device, gt_function=mo_set.gt_function, num_datapoints=5, lower_bound=0, upper_bound=6)
    print(train_X)
    gp_model, optimizer, mll = mo_set.setup_model(train_X=train_X, train_Y=train_Y)
    print(gp_model)

    return gp_model, optimizer, mll, train_X, train_Y, bounds
  
col1, col2, col3, col4, col5 = st.columns([3,1,3,1,3])


with col2:
    if st.button('Initialize'):
        gp_model, optimizer, mll, train_X, train_Y, bounds = initialization_process()
        gp_model = bo_opt.train_loop(gp_model=gp_model, mll=mll, optimizer=optimizer, train_X=train_X, train_Y=train_Y, num_epochs=500)
        model_plot, acq_plot = bo_opt.plot_gp_and_acquisition(model=gp_model, train_X=train_X, train_Y=train_Y, bounds=bounds)
        st.session_state['mll'] = mll
        st.session_state['optimizer'] = optimizer
        st.session_state['model_plot'] = model_plot
        st.session_state['acq_plot'] = acq_plot
        st.session_state['gp_model'] = gp_model
        st.session_state['bounds'] = bounds
        st.session_state['train_X'] = train_X
        st.session_state['train_Y'] = train_Y


with col4:
    if st.button('Iterate'):
        st.session_state['model_plot'], st.session_state['acq_plot'], new_x_value, new_y_value, st.session_state['train_X'], st.session_state['train_Y'], st.session_state['gp_model'] = bo_opt.perform_next_iteration(gt_function=mo_set.gt_function, gp_model=st.session_state['gp_model'], train_X=st.session_state['train_X'], train_Y=st.session_state['train_Y'], bounds=st.session_state['bounds'])
        print(st.session_state['train_Y'])

for _ in range(4):
    st.text("")

col_i1, col_i2 = st.columns([1,1])

with col_i1:
    if st.session_state['model_plot'] is not None:
        st.pyplot(st.session_state['model_plot'])

with col_i2:
    if st.session_state['acq_plot'] is not None:
        st.pyplot(st.session_state['acq_plot'])
