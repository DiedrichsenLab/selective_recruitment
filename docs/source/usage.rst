Usage
=====

Data Extraction
---------------
Cortical and cerebellar data are extracted using Functional_Fusion. 
Cortical surface data is extracted in fsLR space with 32k vertices, and cerebellar data is extracted in SUIT3 (SUIT space 3mm resolution) space.

Using the Functional Fusion toolbox, first create an instance of the WMFS dataset: (see Functional_Fusion documentation for more details on how to create an class for your dataset)

.. code-block:: python

    dataset = ds.DataSetWMFS(base_dir + '/WMFS')

Now to extract cortical data in fs32k space:

.. code-block:: python

    dataset.extract_all(ses_id='ses-01',
                        type='CondAll',
                        atlas='fs32k')


and to extract the data for the cerebellum and smooth it:


.. code-block:: python

    dataset.extract_all(ses_id='ses-01',
                        type='CondAll',
                        atlas='SUIT3', 
                        smooth = 2)

Connectivity Models
-------------------

<to be updated>


Observed and Predicted Cerebellar Data
-------------------------------------- 

data.py contains the code to get the observed data (per voxel/averaged over ROIs) for cerebellar and cortical data. 
Additionally, it has code you can use to get the predicted data from the connectivity models.


For example, say you want to get the observed and predicted data for the cerebellum using a specific connectivity model:
This uses the connectivity model trained on MDTB (all sessions), using the *Icosahedron1002* as the cortical parcellation, 
L2regression, with logalpha = 8, for regularization. 

.. code-block:: python

    cereb_dat, cereb_dat_pred, atlas, info = get_voxdata_obs_pred(dataset = "WMFS",
                                            ses_id = 'ses-02',
                                            subj = None,
                                            atlas_space='SUIT3',
                                            cortex = 'Icosahedron1002',
                                            type = "CondHalf",
                                            add_rest = False,
                                            mname_base = "MDTB_all_Icosahedron1002_L2Regression",
                                            mname_ext = "_A8",
                                            train_type = "train",
                                            crossed = True)


Now you can get the observed and predicted data averaged over voxels within ROIs of a parcellation (say MDTB10):


If you only want to get the data within a specific ROI, you can pass on the name of the ROI (as used in the lut file)
to the roi_selected parameter. If you want to get the data averaged over all ROIs, you can pass on None to the roi_selected parameter.

.. code-block:: python

    obs_df = average_rois(cereb_dat,
                            info,
                            atlas_space = "SUIT3",
                            atlas_roi = "MDTB10",
                            roi_selected = None,
                            unite_struct = False,
                            space = "SUIT", 
                            var = "Y")

And to get the predicted data averaged over ROIs, you can do the same:

.. code-block:: python

    pred_df = average_rois(cereb_dat_pred,
                            info,
                            atlas_space = "SUIT3",
                            atlas_roi = "MDTB10",
                            roi_selected = None,
                            unite_struct = False,
                            space = "SUIT", 
                            var = "X")


Selective recriuitment
----------------------

data.py contains a wrapper function that extracts both the observed and predicted data for a selected cerebellar parcellation and connectivity model.

.. code-block:: python

    df = ss.get_summary_conn(dataset = "WMFS",
                            ses_id = "ses-01",
                            subj = None, # to do all the subjects
                            atlas_space = "SUIT3",
                            cerebellum_roi = "MDTB10",
                            cerebellum_roi_selected = None,
                            cortex_roi = "Icosahedron1002",
                            type = "CondHalf",
                            add_rest = True,
                            mname_base = "MDTB_all_Icosahedron1002_L2Regression", # Fusion_all_Icosahedron1002_L2Regression_05_avg
                            mname_ext = "_A8",
                            crossed = True)


Now you can plot the observed and predicted data for the selected cerebellar parcellation:

.. code-block:: python

    df_roi = df.loc[df['roi_name'] == 'ROI02']


To plot observed vs predicted data within the selected region of interest, you can use the following code from plotting.py:
*label_dict*, *marker_dict*, *color_dict* are dictionaries that map the values of the variable used in *split* to the labels, markers and colors you want to use in the plot.


.. code-block:: python

    make_scatterplot(df_roi, split='cond_num',
                    labels=label_dict,
                    markers=marker_dict,
                    colors=color_dict)


