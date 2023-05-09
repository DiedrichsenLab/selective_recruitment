
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ap(df, 
            parcellation = "NettekovenSym68c32integLR", 
            roi_name = "DR"):

    # get the corresponding part of the dataframe
    DD = df.loc[df.cond_name != 'rest'][df.roi_name == roi_name].copy()
    DD["load"] = DD["load"].apply(str)
    DD['recall'] = DD['recall'].map({1: 'fw', 0: 'bw'})
    DD['phase'] = DD['phase'].map({1: 'Ret', 0: 'Enc'})
    plt.figure()
    sns.catplot(DD, x="load", y="Y", hue="recall",
                col="phase", errorbar='se', kind='point')
    return