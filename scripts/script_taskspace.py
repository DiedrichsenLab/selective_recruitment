import numpy as np
import seaborn as sns # for plots
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix
import selective_recruitment.plotting as spl
import numpy.linalg as la
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
atlas_dir = base_dir + '/Atlases'
wk_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity/'

def get_mdtb_data():
    """Get the task space from the MDTB dataset.
    """
    X,info,myds = ds.get_dataset(base_dir,'MDTB',
                                atlas='fs32k',sess='all',
                                type='CondAll')
    labels, info['cond_num_uni'] = np.unique(info.cond_name,return_inverse=True)
    C = matrix.indicator(info['cond_num_uni'])
    info_new = pd.DataFrame({
            "cond_name": labels,
            "cond_num_uni": np.arange(len(labels))})
    Y= la.pinv(C) @ X # Condense 
    return Y,info_new

def zstandard(X):
    """Get the connectivity matrices from the HCP dataset. 
    """
    Y = X.copy()
    Y = Y - np.mean(Y,axis=1,keepdims=True)
    Y = Y / np.std(Y,axis=1,keepdims=True)
    return Y

def average_roi(Y):
    atlas,ainf = am.get_atlas('fs32k',atlas_dir)
    gii_files = [atlas_dir + '/tpl-fs32k/Icosahedron162.L.label.gii',
                atlas_dir + '/tpl-fs32k/Icosahedron162.R.label.gii']
    label_vec,labels = atlas.get_parcel(gii_files)
    Yn,lab = ds.agg_parcels(Y,label_vec)
    return Yn,lab

def calc_eigenvectors_task():
    Ya, info = get_mdtb_data()
    Y,lab = average_roi(Ya)
    Y = zstandard(Y)

    n_subj,n_cond,n_roi = Y.shape


    U, S, V_task = la.svd(Y.mean(axis=0), full_matrices=False)
    
    with open('eigenvectors_task.npy', 'wb') as f:
        np.save(f, U)
        np.save(f, S)
        np.save(f, V_task)
        np.save(f, Y.mean(axis=0))

    info.to_csv('eigenvector_info.csv')

def calc_eigenvectors_rest():

    subj=np.arange(50)
    COV_list=[]

    for s in subj:
        print(f'Subject {s}')
        Xa,hcp_info,_ = ds.get_dataset(base_dir,'HCP',
                                atlas='fs32k',sess='all',
                                type='Tseries',subj=s)

        X,_ = average_roi(Xa)
        X = zstandard(X)
        _,n_time,n_roi = X.shape

        C = X[0,:,:].T @ X[0,:,:]/n_time
        COV_list.append(C)
    
    C = np.stack(COV_list,axis=0)
    COV = C.mean(axis=0)
    Eig_rest,V_rest = la.eigh(COV)    
    Eig_rest = Eig_rest[::-1]
    V_rest = V_rest[:,::-1]
    
    with open('eigenvectors_rest.npy', 'wb') as f:
        np.save(f, Eig_rest)
        np.save(f, V_rest)
        np.save(f, C)

def load_eigenvectors():

    with open('eigenvectors_task.npy', 'rb') as f:
        U=np.load(f,allow_pickle=True)
        S= np.load(f)
        V_task= np.load(f)
        Y= np.load(f)

    info = pd.read_csv('eigenvector_info.csv')

    with open('eigenvectors_rest.npy', 'rb') as f1:
        Eig_rest=np.load(f1,allow_pickle=True)
        V_rest = np.load(f1,allow_pickle=True)
        C = np.load(f1,allow_pickle=True)

    return U,S,V_task,Y,info,Eig_rest,V_rest,C


def plot_ellipse(mean,COV, ax, n_std=3.0, facecolor='none',**kwargs):
    """
    Plots a ellipse described by mean and covariance

    Args:
    Returns:
    """
    pearson = COV[0, 1]/np.sqrt(COV[0, 0] * COV[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale = np.sqrt(np.diag(COV)) * n_std


    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale[0], scale[1]) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_taskspace(U,S,V_task,C,info,dims=[0,1],ellipse_size=[1,2,3,4]):
    info['X']=U[:,dims[0]]*S[dims[0]]
    info['Y']=U[:,dims[1]]*S[dims[1]]

    # Plot the dimensions 
    sns.scatterplot(info,x='X',y='Y')
    ax= plt.gca()
    ax.axis('equal')


    texts = []
    for i,d in info.iterrows():
        text = plt.text(
                        d.X+0.01,
                        d.Y,
                        s = d.cond_name,
                        horizontalalignment='left',
                        size='small',
                        weight='regular'
                        )
        texts.append(text)

    # Get ellipse of variation in resting-space 
    rest_cov = V_task[dims,:] @ C.mean(axis=0) @ V_task[dims,:].T
    rest_mean = [info.X.iloc[-1],info.Y.iloc[-1]]
    for s in ellipse_size:
        plot_ellipse(rest_mean,rest_cov,ax,n_std=s,facecolor='none',edgecolor='r')


def plot_restspace(Y,V_rest,C,info,dims=[0,1]):
    info['X']=V_rest[:,dims[0]].T @ Y.T
    info['Y']=V_rest[:,dims[1]].T @ Y.T

    # Plot the dimensions 
    sns.scatterplot(info,x='X',y='Y')
    ax= plt.gca()
    ax.axis('equal')

    texts = []
    for i,d in info.iterrows():
        text = plt.text(
                        d.X+0.01,
                        d.Y,
                        s = d.cond_name,
                        horizontalalignment='left',
                        size='small',
                        weight='regular'
                        )
        texts.append(text)

    # Get ellipse of variation in resting-space 
    rest_cov = V_rest[:,dims].T @ C.mean(axis=0) @ V_rest[:,dims]
    rest_mean = [info.X.iloc[-1],info.Y.iloc[-1]]

    plot_ellipse(rest_mean,rest_cov,ax,n_std=0.1,facecolor='r',edgecolor='r',alpha=0.2)
    

if __name__=='__main__':
    # calc_eigenvectors_rest()
    U,S,V_task,Y,info,Eig_rest,V_rest,C= load_eigenvectors()
    plot_taskspace(U,S,V_task,C,info,dims=[0,1],ellipse_size=[0.75,1.5,3,6])
    # plot_restspace(Y,V_rest,C,info,dims=[0,2])
    # calc_eigenvectors_rest()
    plt.savefig('taskspace.pdf',dpi=300)
    pass