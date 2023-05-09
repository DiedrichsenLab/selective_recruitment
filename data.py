
"""Make Universal..... """
def get_voxeldata_and_prediction(ses_id = 'ses-02',
                subj = None,
                atlas_space='SUIT3',
                cortex = 'Icosahedron1002',
                type = "CondAll",
                mname = "MDTB_ses-s1_Icosahedron1002_L2Regression",
                reg = "A8",
                add_rest = False):
    """_summary_

    Args:
        ses_id (str): _description_. Defaults to 'ses-02'.
        subj (array or str or None): Subjects. None = all 
        atlas_space (str): cerebellar atlas space. Defaults to 'SUIT3'.
        cortex (str, optional):cortical parcellation Defaults to 'Icosahedron1002'.
        type (str): Data Type. Defaults to "CondAll".
        mname (str): connectivity model name Defaults to "MDTB_ses-s1_Icosahedron1002_L2Regression".
        reg (str): Regularization string. Defaults to "A8".
        add_rest (bool): Add rest to data? Defaults to False.
    Returns:
        Y: cerebellar data
        YP: predicted cerebellar data
        atlas: cortical atlas
        info: dataframe with info for data  
    Returns:
    """
    Y,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas=atlas_space,
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    X,info,dset = ds.get_dataset(gl.base_dir,'WMFS',
                                    atlas="fs32k",
                                    sess=ses_id,
                                    subj=subj,
                                    type = type)
    model_path = os.path.join(ccc_gl.conn_dir,atlas_space,'train',mname)
    fname = model_path + f"/{mname}_{reg}_avg.h5"
    json_name = model_path + f"/{mname}_{reg}_avg.json"
    conn_model = dd.io.load(fname)

    atlas,ainf = am.get_atlas('fs32k',gl.atlas_dir)
    label=[gl.atlas_dir+'/tpl-fs32k/'+cortex+'.L.label.gii',
           gl.atlas_dir+'/tpl-fs32k/'+cortex+'.R.label.gii']
    atlas.get_parcel(label,unite_struct=False)
    X, parcel_labels = ds.agg_parcels(X , 
                                         atlas.label_vector, 
                                         fcn=np.nanmean)
    YP = conn_model.predict(X)
    if add_rest:
        Y,_ = ra.add_rest_to_data(Y,info)
        YP,info = ra.add_rest_to_data(YP,info)

    return Y,YP,atlas,info

# Should all voxel-wise, then aggregate to parcels

def get_summary_conn(dataset = "WMFS", 
                     ses_id = 'ses-02', 
                     atlas_space = "SUIT3", 
                     cerebellum_roi = "Verbal2Back", 
                     cortex_roi = "Icosahedron1002",
                     type = "CondHalf", 
                     add_rest = True,
                     conn_dataset = "MDTB",
                     conn_method = "L2Regression", 
                     log_alpha = 8, 
                     crossed = True, 
                     conn_ses_id = "ses-s1"):

    """
    Function to get summary dataframe using connectivity model to predict cerebellar activation.
    It's written similar to get_symmary from recruite_ana code
    """
    
    tensor_cerebellum, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas=atlas_space,sess=ses_id,type=type, info_only=False)
    tensor_cortex, info, _ = fdata.get_dataset(gl.base_dir,dataset,atlas="fs32k",sess=ses_id,type=type, info_only=False)

    # get connectivity weights and scaling 
    conn_dir = gl.conn_dir + f"/{atlas_space}/train/{conn_dataset}_{conn_ses_id}_{cortex_roi}_{conn_method}"

    # load the model averaged over subjects
    fname = conn_dir + f"/{conn_dataset}_{conn_ses_id}_{cortex_roi}_{conn_method}_A{log_alpha}_avg.h5"
    model = dd.io.load(fname) 
    # weights = model.coef_
    # scale = model.scale_ 

    # prepare the cortical data 
    # NOTE: to use connectivity weights estimated in MDTB you always need to pass a tesselation
    # BECAUSE the models have been trained using tesselations
    cortex_label = []
    for hemi in ['L', 'R']:
        cortex_label.append(gl.atlas_dir + '/tpl-fs32k' + f'/{cortex_roi}.{hemi}.label.gii')
    X_parcel, ainfo, X_parcel_labels = ra.agg_data(tensor_cortex, "fs32k", cortex_label, unite_struct = False)

    # use cortical data to predict cerebellar data (voxel-wise)
    atlas_cereb, _ = am.get_atlas("SUIT3",gl.atlas_dir)
    # Yhat = ra.predict_cerebellum(weights, scale, X_parcel, atlas_cereb, info, fwhm = 0)
    if crossed:
        X_parcel = np.concatenate([X_parcel[:, info.half == 2, :], X_parcel[:, info.half == 1, :]], axis=1)
    Yhat = model.predict(X_parcel)
    # get the cerebellar data
    # NOTE: if None is passed, then it will average over the whole cerebellum
    if cerebellum_roi is not None:
        cerebellum_label = gl.atlas_dir + '/tpl-SUIT' + f'/atl-{cerebellum_roi}_space-SUIT_dseg.nii'
        # use lookuptable to get region info
        region_info = sroi.get_label_names(cerebellum_roi) 
        # get observed cerebellar data
        Y_parcel, ainfo, Y_parcel_labels = ra.agg_data(tensor_cerebellum, "SUIT3", cerebellum_label, unite_struct = False)

        # get predicted cerebellar data
        Yhat_parcel, ainfo, Yhat_parcel_labels = ra.agg_data(Yhat, "SUIT3", cerebellum_label, unite_struct = False)

    else: 
        # there's only one parcel: the whole cerebellum
        parcels = [1]
        # aggregate observed values over the whole cerebellum
        # aggregate predicted values over the whole cerebellum
        pass

    # add rest condition for control?
    if add_rest:
        Yhat_parcel,_ = ra.add_rest_to_data(Yhat_parcel,info)
        Y_parcel,info = ra.add_rest_to_data(Y_parcel,info)
        
    # Transform into a dataframe with Yhat and Y data 
    n_subj,n_cond,n_roi = Yhat_parcel.shape

    summary_list = [] 
    for i in range(n_subj):
        for r in range(n_roi):
            info_sub = info.copy()
            vec = np.ones((len(info_sub),))
            info_sub["sn"]    = i * vec
            info_sub["roi"]   = Yhat_parcel_labels[r] * vec
            info_sub["roi_name"] = region_info[r+1]
            info_sub["X"]     = Yhat_parcel[i,:,r]
            info_sub["Y"]     = Y_parcel[i,:,r]

            summary_list.append(info_sub)
        
    summary_df = pd.concat(summary_list, axis = 0,ignore_index=True)
    return summary_df

