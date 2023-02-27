# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from pathlib import Path
import numpy as np
# from util import *
import Functional_Fusion.dataset as fdata
import Functional_Fusion.atlas_map as am
import SUITPy.flatmap as flatmap
from nilearn import plotting
# from surfplot import Plot
import nitools as nt

import matplotlib.pyplot as plt
import nibabel as nb


# Import Dash dependencies
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from PIL import Image

#TODO: overlay two maps using suit flatmap
#TODO: use dash to plot contrast + roi map as overlay (with transparency)
#TODO: click on a cerebellar voxel/region and get the connectivity map on the cortex

# set base directory of the functional fusion 
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'
conn_dir = '/cifs/diedrichsen/data/Cerebellum/connectivity/'

# create an instance of WMFS class
Dataset = fdata.get_dataset_class(base_dir, dataset= "WMFS")
# get info 
info = Dataset.get_info("ses-02","CondAll")
info['name'] = info['cond_name'].map(lambda x: x.rstrip('  '))

# get participants
subjects = list(Dataset.get_participants().participant_id.values)
subjects.append("group")
conditions = list(np.unique(info["name"]))

# get cortical surfaces for plotting
surfs = [atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

flat = flatmap._surf_dir + f'/FLAT.surf.gii'

#start of app
app = Dash(__name__)

app.layout = html.Div([ 
    html.Div(
        [
            html.H2('Cerebellar activation'),
            html.Div([
                dcc.Graph(id="figure_cerebellum", style={"height":"1000", "width":"1000"}, 
                          clear_on_unhover=False),
                ], 
            ),
            html.H2('Cortical activation'),
            html.Div([
                dcc.Graph(id="figure_cortex", style={"height":"1000", "width":"1000"}, 
                          clear_on_unhover=False),
                ],
            ),
        ], 
    ),
        
    html.Div(
        [
            html.Div(children=[
                html.Label('Condition'),
                dcc.Dropdown(conditions, id='chosen_condition',
                             value=conditions[0], clearable=False),
                ], style={'padding': 10, 'flex': 1}
            ),

            html.Div(children = [
                html.Label('Subject'), 
                dcc.Dropdown(subjects, id = 'chosen_subject', 
                             value = subjects[0], clearable = False)
                ], style={'padding': 10, 'flex': 1}
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'font-family':"Calibri (Body)"}),

    ], style={'display': 'flex', 'flex-direction': 'column', 'flex': 300}
)

def plot_cerebellum_flatmap(subject, contrast_name, cmap = "coolwarm"):
    """
    plot the contrast on a flatmap
    """
    path_to_cifti = Dataset.data_dir.format(subject) + f'/{subject}_space-SUIT3_ses-02_CondAll.dscalar.nii'
    cifti_img = nb.load(path_to_cifti)

    # get the numpy array 
    img = cifti_img.get_fdata()
    # get info for the contrast
    info_con = info.loc[info.names == contrast_name]
    idx = info_con.reg_id.values[0]

    # get the numpy array corresponding to the contrast
    img_con = img[idx-1, :].reshape(-1, 1)

    # convert vol 2 surf
    atlas_cereb, ainfo = am.get_atlas('SUIT3',atlas_dir)
    img_nii = atlas_cereb.data_to_nifti(img_con.T)
    img_flat = flatmap.vol_to_surf([img_nii], stats='nanmean', space = 'SUIT')

    # plot 
    # fig = plotting.plot_surf_stat_map(
    #                                     flat, img_flat, hemi='right',
    #                                     # title='Surface left hemisphere',
    #                                     colorbar=True, 
    #                                     view = 'medial',
    #                                     cmap="coolwarm",
    #                                     engine='plotly',
    #                                 )

    # ax = fig.figure
    ax = flatmap.plot(data=img_flat, render="plotly", hover='auto', cmap = cmap, colorbar = True, bordersize = 1, cscale = (-0.2, 0.2))
    return ax

def plot_cortex_surface(subject, contrast_name, cmap = "coolwarm"):
    """
    """
    # get left and right hemi data into a list
    path_to_cifti = Dataset.data_dir.format(subject) + f'/{subject}_space-fs32k_ses-02_CondAll.dscalar.nii'
    cifti = nb.load(path_to_cifti)
    dat_list = nt.surf_from_cifti(cifti)

    # get info for the contrast
    info_con = info.loc[info.names == contrast_name]
    idx = info_con.reg_id.values[0]

    # get the numpy array corresponding to the contrast
    img_con_list = [dat_list[i][idx-1, :].reshape(-1, 1) for i, h in enumerate(['L', 'R'])]

    

    fig = plotting.plot_surf_stat_map(
                                        surfs[0], img_con_list[0], hemi='left',
                                        # title='Surface left hemisphere',
                                        colorbar=True, 
                                        view = 'lateral',
                                        cmap="coolwarm",
                                        engine='plotly',
                                        symmetric_cbar = True,
                                        vmax = 1
                                    )

    ax = fig.figure

    return ax

# callback for cerebellar map
@app.callback(
    Output(component_id='figure_cerebellum', component_property='figure'),
    Input(component_id='chosen_subject', component_property='value'),
    Input(component_id='chosen_condition', component_property='value'),
    prevent_initial_call=True,
    )

def show_cerebellum(subject, condition):
    ax1 = plot_cerebellum_flatmap(subject, condition, cmap = "coolwarm")
    return ax1

# callback for cortical map
@app.callback(
    Output(component_id='figure_cortex', component_property='figure'),
    Input(component_id='chosen_subject', component_property='value'),
    Input(component_id='chosen_condition', component_property='value'),
    prevent_initial_call=True,
    )

def show_cortex(subject, condition):
    ax2 = plot_cortex_surface(subject, condition, cmap = "coolwarm")
    return ax2


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8080)