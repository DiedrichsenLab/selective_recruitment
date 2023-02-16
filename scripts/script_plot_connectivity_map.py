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
import os

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
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'
conn_dir = '/srv/diedrichsen/data/Cerebellum/connectivity/'

# create an instance of WMFS class
Dataset = fdata.get_dataset_class(base_dir, dataset= "WMFS")
# get info 
info = Dataset.get_info("ses-02","CondAll")
info['name'] = info['cond_name'].map(lambda x: x.rstrip('  '))

# get participants
subjects = list(Dataset.get_participants().participant_id.values)
subjects.append("group")

parcellations = ["MDTB10", "NettekovenSym34"]

# get cortical surfaces for plotting
surfs = [atlas_dir + f'/tpl-fs32k/tpl_fs32k_hemi-{h}_inflated.surf.gii' for i, h in enumerate(['L', 'R'])]

flat = flatmap._surf_dir + f'/FLAT.surf.gii'

#start of app
app = Dash(__name__)
click_region_labels = dcc.Markdown(id='clicked-parcel')

app.layout = html.Div([ 
    html.Div(
        [
            html.H2('Cerebellar parcellation'),
            html.Div([
                dcc.Graph(id="figure_cerebellum", style={"height":"1000", "width":"1000"}, 
                          clear_on_unhover=False),
                ], 
            ),
            html.H2('Average connectivity weight'),
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
                html.Label('parcellation'),
                dcc.Dropdown(parcellations, id='chosen_parcellation',
                             value=parcellations[0], clearable=False),
                ], style={'padding': 10, 'flex': 1}
            ),
        ], style={'width': '40%', 'display': 'inline-block', 'font-family':"Calibri (Body)"}),
    html.Div(
        [
            html.H4(id='clicked-parcel')
        ]
    )

    ], style={'display': 'flex', 'flex-direction': 'column', 'flex': 300}
)

def plot_cerebellum_flatmap(parcellation, cmap = "coolwarm"):
    """
    plot the contrast on a flatmap
    """
    path_to_img = atlas_dir+ f'/tpl-SUIT/atl-{parcellation}_dseg.label.gii'
    img_nii = nb.load(path_to_img)

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
    ax = flatmap.plot(data=img_nii, render="plotly", hover='auto', colorbar = False, bordersize = 1, overlay_type='label')
    return ax, img_nii

def plot_cortex_surface(parcellation, parcel):
    """
    """
    
    # When initiliazing the website and if clickin on a null region, show no conditions
    
    # get the file containing best weights
    filename = os.path.join(conn_dir, "MDTB", 'train', f'Icosahedron-1002_Sym.32k_{parcellation}_L2Regression_8.dscalar.nii')
    cifti = nb.load(filename)
    dat_list = nt.surf_from_cifti(cifti)

    # get the map corresponding to the parcel
    if parcel is None:
        parcel = 1
    else:
        map_parcel_list = [dat_list[h][:, parcel] for h, hemi in enumerate(['L', 'R'])]
    

    fig = plotting.plot_surf_stat_map(
                                        surfs[0], map_parcel_list[0], hemi='left',
                                        # title='Surface left hemisphere',
                                        colorbar=True, 
                                        view = 'lateral',
                                        cmap="viridis",
                                        engine='plotly',
                                        symmetric_cbar = True,
                                        vmax = 0.0008
                                    )

    ax = fig.figure

    return ax

def get_clicked_parcel():
    return

# callback for cerebellar map
@app.callback(
    Output(component_id='figure_cerebellum', component_property='figure'),
    Input(component_id='chosen_parcellation', component_property='value'),
    prevent_initial_call=True,
    )

def show_cerebellum(parcellation):
    ax1, _ = plot_cerebellum_flatmap(parcellation)
    return ax1

# callback for cortical map
@app.callback(
    Output(component_id='figure_cortex', component_property='figure'),
    Input(component_id='chosen_parcellation', component_property='value'),
    Input(component_id='figure-cerebellum', component_property='clickData'),
    prevent_initial_call=True,
    )

def show_cortex(parcellation, parcel):
    ax2 = plot_cortex_surface("MDTB10", 1)
    _, img_nii = plot_cerebellum_flatmap(parcellation)
    # get data
    d = img_nii.agg_data()
    if (parcel is not None) and (sum(d == parcel) != 0):
        
        ax2 = plot_cortex_surface(parcellation, parcel, cmap = "coolwarm")
    return ax2


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.2', port=8080)