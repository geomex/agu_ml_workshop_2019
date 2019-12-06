'''

This utilities script is intended for the students attending the Early Career Scientist Machine Learning Workshop at the American Geophysical Union Conference 2019. 

Joel A. Gongora
Data Scientist
PhD Candidate
Boise State University

email:  geodatamex@gmail.com
        joel.gongora@elderresearch.com
'''

import copy
import gzip
import pickle
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d
from sklearn.preprocessing import StandardScaler

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def df_to_array(
    datos=None
    ,NROWS=None
    ,NCOLS=None
    ,coords=None
    ,omit=None
    ,output=None
    ,order='C'
):
    '''
    
    Description: Converts a dataframe into a numpy 
    arrays X, Y, xcoords, ycoords. 
    
    ------------------------
    Inputs:
    datos - (dataframe) 
    NROWS - (int) numer of rows in array: X
    NCOLS - (int) number of columns in the array: X
    coords - (list strings) column names of the coordinates
    omit -- (list strings) columns in data frame to omit from the features
    output - (list strings) column name of the target [Y]
    
    ------------------------
    
    Outputs:
    X -- (np.array, dim=[NROWS x NCOLS x no_features])
    Y -- (np.array [NROWS x NCOLS x no_outputs])
    UTME -- (np.array [NROWS x NCOLS])
    UTMN -- (np.array [NROWS x NCOLS]) 
    features - (list strings) names of the features w.r.t the arrays
    
    -----------------------
    
    '''
    X = np.empty(
        (NROWS, NCOLS, len(datos.columns) - len(coords+omit) - len(output))
    )

    Y = np.empty(
        (NROWS, NCOLS, 1)
    )

    UTMN = np.empty(
        (NROWS, NCOLS, 1)
    )

    UTME = np.empty(
        (NROWS, NCOLS, 1)
    )

    ix = 0
    ixx = 0
    iy = 0
    features = []
    for col in datos.columns.tolist():
        if col in output:
            Y[:, :, iy] = datos[col].values.reshape(
                NROWS, NCOLS, order=order
            )
            iy = iy + 1
        elif col not in coords + omit:
            X[:, :, ix] = datos[col].values.reshape(
                NROWS, NCOLS, order=order
            )
            ix = ix + 1
            features.append(col)
        elif col in coords[0]:
            UTME[:, :, ixx] = datos[col].values.reshape(
                NROWS, NCOLS, order=order
            )
        elif col in coords[1]:
            UTMN[:, :, ixx] = datos[col].values.reshape(
                NROWS, NCOLS, order=order
            )
    return X, Y, UTME, UTMN, features



def model_assessor(X, Y, loaded_model, val_index):
    
    '''
    Inputs:
    
    Outputs:
    
    '''
    
    Ycal = {}
    rsquared = {}
    mae = {}
    pearson = {}
    rmse = {}
    
    for item in ['train','test','val']:
        rsquared[item] = []
        mae[item] = []
        pearson[item] = []
        rmse[item] = []
    
    Ycal['test'] = loaded_model.predict(X['test']).flatten()
    Ycal['test'][Ycal['test'] < 0] = 0
    Ycal['val'] = loaded_model.predict(X['train'][val_index, :, :, :])
    Ycal['val'][Ycal['val'] < 0 ] = 0
    Ycal['train'] = loaded_model.predict(
        np.delete(X['train'], val_index, axis=0)
    )
    Ycal['train'][Ycal['train'] < 0] = 0

    # Calculated Rsquared

    rsquared['train'].append(
        metrics.r2_score(
            np.delete(Y['train'], val_index, axis=0).flatten()
            ,Ycal['train'].flatten(),
        )
    )

    rsquared['val'].append(
        metrics.r2_score(
            Y['train'][val_index, :, :].flatten()
            ,Ycal['val'].flatten(),
        )
    )
    rsquared['test'].append(
        metrics.r2_score(Y['test'].flatten(), Ycal['test'].flatten())
    )

    # Calculate Pearson Correlation


    pearson['train'].append(    
        pearsonr(
            np.delete(Y['train'], val_index, axis=0).flatten()
            ,Ycal['train'].flatten())[0]
    )

    pearson['val'].append(
        pearsonr(
            Y['train'][val_index, :, :, :].flatten()
            ,Ycal['val'].flatten())[0]
    )

    pearson['test'].append(
        pearsonr(Y['test'].flatten(), Ycal['test'].flatten())[0]
    )

    # Calculate Mean Absolute Error

    mae['train'].append(
        metrics.mean_absolute_error(
            Ycal['train'].flatten()
            ,np.delete(Y['train'], val_index, axis=0).flatten()
        )
    )

    mae['val'].append(
        metrics.mean_absolute_error(
            Ycal['val'].flatten()
            ,Y['train'][val_index, :, :, :].flatten()
        )
    )

    mae['test'].append(
        metrics.mean_absolute_error(
            Ycal['test'].flatten(), Y['test'].flatten()
        )
    )

    # Calculate Root Mean Squared Error
    rmse['train'].append(
        np.sqrt(
            metrics.mean_squared_error(
                Ycal['train'].flatten()
                ,np.delete(Y['train'], val_index, axis=0).flatten()
            )
        )
    )

    rmse['val'].append(
        np.sqrt(
            metrics.mean_squared_error(
                Ycal['val'].flatten()
                ,Y['train'][val_index, :, :, :].flatten()
            )
        )
    )

    rmse['test'].append(
        np.sqrt(
            metrics.mean_squared_error(
                Ycal['test'].flatten(), Y['test'].flatten()
            )
        )
    )
    
    model_assessment = pd.concat(
    [
        pd.DataFrame.from_dict(
            rsquared
            ,orient='columns'
        )[['train','val','test']].T.rename(columns={0:'R^2'})
    
        ,pd.DataFrame.from_dict(
            rmse
            ,orient='columns'
        )[['train','val','test']].T.rename(columns={0:'RMSE'})
    
        ,pd.DataFrame.from_dict(
            mae
            ,orient='columns'
        )[['train','val','test']].T.rename(columns={0:'MAE'})
    
        ,pd.DataFrame.from_dict(
            pearson
            ,orient='columns'
        )[['train','val','test']].T.rename(columns={0:'Pearson'})
    ]
    ,axis=1
)
    
    return model_assessment



def plot_features(data=None, features=None, UTME=None, UTMN=None):
    # ------------------------------------ #
    # Use mplstyle for consistant plotting #
    # ------------------------------------ #

    plt.style.use(
        '../plotstyles/nolatex_smallfont.mplstyle'
    )

    # ------------------------- #
    # Iterate through each site #
    # ------------------------- #

    for idx, site in enumerate(data.keys()):
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=0.35)
        print('\n')

        # ---------------------------- #
        # Iterate through each Feature #
        # ---------------------------- #

        for feat_idx, feat in enumerate(features[site]):
            # -------------------------------------- #
            # Define specific colormaps for features #
            # -------------------------------------- #
            if feat in ['aspect']:
                cmap = 'hsv'
            elif feat in ['veg_h','raw_off','tbreak_45']:
                cmap = 'gray'
            else:
                cmap = 'jet'
            # ---------------------------------------------- #
            # Apply Median Filtering to a Subset of Features #
            # ---------------------------------------------- #
            if feat not in ['raw_off', 'snow_off']:
                valores = medfilt2d(
                        np.flipud(X[site][:,:,feat_idx])
                        ,kernel_size=7
                    )
            else:
                valores = np.flipud(X[site][:,:,feat_idx])
            # ---------------------------------------------- #
            # Make Use of Subplot to Create A [2x3] grid     #
            # and pass to variable "ax" for future reference #
            # ---------------------------------------------- #            
            ax = plt.subplot(2,3,feat_idx+1)

            # ------------------------------------------------ #
            # Use matplotlib.pyplot.imshow() to plot the image #
            # ------------------------------------------------ #

            im = plt.imshow(
                extent=[
                    UTME[site].min()
                    ,UTME[site].max()
                    ,UTMN[site].min()
                    ,UTMN[site].max()
                ]
                ,X=valores
                ,cmap=cmap
            )

            if feat_idx < len(features[site]) - 1:
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
            else:
                plt.xlabel('UTME')
                plt.ylabel('UTMN')


            plt.title(feat, fontsize=10)
            if feat in ['slope']:
                plt.clim([0,1])

            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
            for tick in ax.get_yticklabels():
                tick.set_rotation(30)

            if feat_idx == 0:
                print(
                    "--------------------------------------------"
                    + "--------------------------------\n"
                    "\t\t\t\t"
                    + " ".join([s.upper() for s in site.split('_')])
                    + "\n"
                )
        plt.show()
        
        
def plot_ex_pred_vs_observed(
    scaled_train_test=None
    ,prediction_img=None
    ,index_to_predict=None
    ,site=None
    ,train_or_test=None
):
    plt.style.use(
        '../plotstyles/nolatex_smallfont.mplstyle'
    )

    fig, ax = plt.subplots(1,2,figsize=(10, 10))

    # --------------- #
    # Plot the Output #
    # --------------- #

    ax[1].imshow(
        np.flipud(prediction_img)
        ,extent=[
            scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    );

    # --------------- #
    # Set the Title  #
    # --------------- #

    ax[1].set_title(
        'Predicted'
        ,fontsize = 18
    )

    # --------------- #
    # Plot the Output #
    # --------------- #

    ax[0].imshow(
        np.flipud(scaled_train_test['Y'][site][train_or_test][index_to_predict].squeeze(axis=-1))
        ,extent=[
            scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    )

    # --------------- #
    # Set the Title   #
    # --------------- #

    ax[0].set_title(
        'Observed'
        ,fontsize = 18
    );

    for idx in [0, 1]:
        for tick in ax[idx].get_xticklabels():
            tick.set_rotation(30)
        for tick in ax[idx].get_yticklabels():
            tick.set_rotation(30)
    # plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    fig.get_axes()[0].annotate(
        " ".join([s.upper() for s in site.split('_')]) + ':  ' + train_or_test.upper()
        ,(0.5, 0.85)
        ,xycoords='figure fraction'
        ,ha='center'
        ,fontsize=24
    );
    plt.show()
    
    
    
def plot_perturb_ex(
    prediction_img=None
    ,permuted_prediction_img=None
    ,scaled_train_test=None
    ,permuted_train_test=None
    ,index_to_predict=None
    ,site=None
    ,train_or_test=None
):

    plt.style.use(
        '../plotstyles/nolatex_smallfont.mplstyle'
    )

    fig, ax = plt.subplots(2,2,figsize=(10, 10))

    # --------------- #
    # Plot the Output #
    # --------------- #

    ax[0,1].imshow(
        np.flipud(prediction_img)
        ,extent=[
            scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    );

    # --------------- #
    # Set the Title  #
    # --------------- #


    ax[0,1].set_title(
        'Predicted'
        ,fontsize = 18
    )

    # --------------- #
    # Plot the Output #
    # --------------- #


    ax[0,0].imshow(
        np.flipud(scaled_train_test['Y'][site][train_or_test][index_to_predict].squeeze(axis=-1))
        ,extent=[
            scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    )

    # --------------- #
    # Set the Title   #
    # --------------- #

    ax[0,0].set_title(
        'Observed'
        ,fontsize = 18
    );

    for idx in [0, 1]:
        for tick in ax[0,idx].get_xticklabels():
            tick.set_rotation(30)
        for tick in ax[0,idx].get_yticklabels():
            tick.set_rotation(30)

    # --------------- #
    # Plot the Output #
    # --------------- #

    ax[1,1].imshow(
        np.flipud(permuted_prediction_img)
        ,extent=[
            permuted_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,permuted_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,permuted_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,permuted_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    );

    # --------------- #
    # Set the Title  #
    # --------------- #

    ax[1,1].set_title(
        'Permuted Predicted'
        ,fontsize = 18
    )

    # --------------- #
    # Plot the Output #
    # --------------- #


    ax[1,0].imshow(
        np.flipud(scaled_train_test['Y'][site][train_or_test][index_to_predict].squeeze(axis=-1))
        ,extent=[
            scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTME'][site][train_or_test][index_to_predict,:,:].max()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].min()
            ,scaled_train_test['UTMN'][site][train_or_test][index_to_predict,:,:].max()
        ]

    )

    # --------------- #
    # Set the Title   #
    # --------------- #

    ax[1,0].set_title(
        'Observed'
        ,fontsize = 18
    );

    for idx in [0, 1]:
        for tick in ax[1, idx].get_xticklabels():
            tick.set_rotation(30)
        for tick in ax[1, idx].get_yticklabels():
            tick.set_rotation(30)

    plt.tight_layout()

    
    
def plot_outputs(
    data=None
    ,site=None
    ,UTME=None
    ,UTMN=None
    ,Y=None
    ,med_filt=False
    ,kernel_size=0
):
    # ------------------------------------ #
    # Use mplstyle for consistant plotting #
    # ------------------------------------ #

    plt.style.use(
        '../plotstyles/nolatex_smallfont.mplstyle'
    )

    fig, ax = plt.subplots(1,2,figsize=(10, 10))

    # ---------------------------- #
    # Iterate through each Site    #
    # ---------------------------- #

    for idx, site in enumerate(data.keys()):

        # ------------------------- #
        # Set the Title of the Site #
        # ------------------------- #

        ax[idx].set_title(
            " ".join([s.upper() for s in site.split('_')])
            ,fontsize = 12
        )

        # ------------------------- #
        # Add white pace to plots   #
        # ------------------------- #

        plt.subplots_adjust(wspace=0.75)

        # ------------------------- #
        # Plot the Image            #
        # ------------------------- #
        if med_filt==False:
            im = ax[idx].imshow(
                extent=[
                    UTME[site].min()
                    ,UTME[site].max()
                    ,UTMN[site].min()
                    ,UTMN[site].max()
                ]
                ,X=np.flipud(
                        Y[site].squeeze(axis=-1)
                )
            )
        else:
            im = ax[idx].imshow(
                extent=[
                    UTME[site].min()
                    ,UTME[site].max()
                    ,UTMN[site].min()
                    ,UTMN[site].max()
                ]
                ,X=np.flipud(
                    medfilt2d(
                        Y[site].squeeze(axis=-1)
                        ,kernel_size = kernel_size
                    )
                )
            )
    
        # ------------------------- #
        # Adjust the Colorbar       #
        # ------------------------- #

        divider = make_axes_locatable(ax[idx])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')

        # ------------------------- #
        # Add Label to Colorbar     #
        # ------------------------- #    

        cbar.set_label('Snow Depth [m]')

        # ------------------------- #
        # Rotate X and Y Ticks      #
        # ------------------------- #        

        for tick in ax[idx].get_xticklabels():
            tick.set_rotation(30)
        for tick in ax[idx].get_yticklabels():
            tick.set_rotation(30)

        # ------------------------- #
        # Reduce White Space        #
        # ------------------------- #         

        plt.tight_layout()

        
def windowing_tool(
    signal=None, kernel_size=(100,100), stride=(20,20)
):
    
    '''
    Inputs:
    
        signal - (np.array, shape=[nrows, ncols, nfeatures])
        kernel_size - (tuple) size of moving window
        stride - (tuple) row and column stride
        
    Outputs:
        
        sample - (n_windows, kernel_size, nfeatures)
    
    '''
    sample = []

    start_y = 0
    end_y = kernel_size[0]
    start_x = 0
    end_x = kernel_size[1]

    # First move through the y-axis ------------------

    while (end_y <= signal.shape[0]):

        while end_x <= signal.shape[1]:
            sample.append(
                signal[start_y:end_y, start_x:end_x, :]
            )

            # Increment X direction by stride -------------

            start_x = start_x + stride[1]
            end_x = end_x + stride[1]

        # Increment Y direction by stride --------------

        start_y = start_y + stride[0]
        end_y = end_y + stride[0]

        # Restart the X Location and Stride ------------------

        start_x = 0
        end_x = kernel_size[1]

    return sample

def plot_output_feature(
    site=None
    ,features=None
    ,no_feat=None
    ,img_index=None
    ,stacked_arrays=None
):
    
    '''
    Inputs:
    
    site -- (string) site_name
    features -- (list) list of features
    no_feature - (int) index of feature from list of features
    
    Ouputs:
    
    plot.
    
    '''
    
    plt.style.use(
        '../plotstyles/nolatex_smallfont.mplstyle'
    )

    fig, ax = plt.subplots(1,2,figsize=(10, 10))

    # --------------- #
    # Plot the Output #
    # --------------- #

    ax[0].imshow(
        np.flipud(stacked_arrays['Y'][site][img_index,:,:,0])
        ,extent=[
            stacked_arrays['UTME'][site][img_index,:,:,0].min()
            ,stacked_arrays['UTME'][site][img_index,:,:,0].max()
            ,stacked_arrays['UTMN'][site][img_index,:,:,0].min()
            ,stacked_arrays['UTMN'][site][img_index,:,:,0].max()
        ]

    );

    # --------------- #
    # Set the Title  #
    # --------------- #


    ax[0].set_title(
        'Snow Depth'
        ,fontsize = 12
    )

    # --------------- #
    # Plot the Output #
    # --------------- #


    ax[1].imshow(
        np.flipud(stacked_arrays['X'][site][img_index,:,:,no_feat])
        ,extent=[
            stacked_arrays['UTME'][site][img_index,:,:,0].min()
            ,stacked_arrays['UTME'][site][img_index,:,:,0].max()
            ,stacked_arrays['UTMN'][site][img_index,:,:,0].min()
            ,stacked_arrays['UTMN'][site][img_index,:,:,0].max()
        ]

    )

    # --------------- #
    # Set the Title   #
    # --------------- #

    ax[1].set_title(
        features[site][no_feat]
        ,fontsize = 12
    );
    for idx in [0, 1]:
        for tick in ax[idx].get_xticklabels():
            tick.set_rotation(30)
        for tick in ax[idx].get_yticklabels():
            tick.set_rotation(30)

            
            
            
            
def scale_data(
    X=None
    ,Y=None
    ,UTME=None
    ,UTMN=None
    ,train_per=0.1
    ,seed='default'
):
    Xout = {}
    Yout = {}
    UTMEout = {}
    UTMNout = {}
    ixs = {}

    # ---------------------------------------- #
    # If we Pass in a List of Training Samples #
    # ---------------------------------------- #
    
    if isinstance(train_per, list): 
        train_size = [int(i) for i in train_per]
        ixs['train'] = train_size
        ixs['test'] = [
            val 
            for val in np.arange(X.shape[0])
            if val not in train_size
        ]    

    # -------- #
    # Permuter #
    # -------- #
    
    elif train_per < 1:
        # ----------------------------- #
        # If we pass a value less than 1 #
        # ----------------------------- #
        train_size = np.int(np.ceil(train_per * X.shape[0]))
        np.random.seed(seed)
        permutation = np.random.permutation(X.shape[0])
        ixs['train'] = permutation[:train_size]
        ixs['test'] = permutation[train_size:]   
    else:
        # --------------------------------------------------------- #
        # If we pass an integer for the number of training examples #
        # --------------------------------------------------------- #
        train_size = train_per
        np.random.seed(seed)
        permutation = np.random.permutation(X.shape[0])
        ixs['train'] = permutation[:train_size]
        ixs['test'] = permutation[train_size:]            

    # Trainset ------------------------
    Xout['train'] = X[ixs['train'], :, :, :]
    Yout['train'] = Y[ixs['train'], :, :]
    UTMEout['train'] = UTME[ixs['train'], :, :, :]
    UTMNout['train'] = UTMN[ixs['train'], :, :, :]

    # Testset -------------------------
    Xout['test'] = X[ixs['test'], :, :, :]
    Yout['test'] = Y[ixs['test'], :, :]

    UTMEout['test'] = UTME[ixs['test'], :, :, :]
    UTMNout['test'] = UTMN[ixs['test'], :, :, :]

    # Scale ------------------
    scaler = StandardScaler()
    Xtr_tmp = scaler.fit_transform(Xout['train'].reshape(-1, X.shape[-1]))
    Xout['train'] = Xtr_tmp.reshape(Xout['train'].shape)
    Xte_tmp = scaler.transform(Xout['test'].reshape(-1, X.shape[-1]))
    Xout['test'] = Xte_tmp.reshape(Xout['test'].shape)
    
    return Xout, Yout, UTMEout, UTMNout




def feature_importance(
    scaled_train_test=None
    ,model=None
    ,features=None
    ,site=None
    ,val_index=None
):
    
    # ------------------ #
    # Initialize Output  #
    # ------------------ #    

    output = pd.DataFrame(
        index=features[site]
        ,columns=['R^2', 'RMSE', 'MAE', 'Pearson']
    )
    
    for f_idx, feat in enumerate(features[site]): 
        
        # --------------------------- #
        # Copy Scaled Train and Test  #
        # --------------------------- # 
        
        permuted_train_test = copy.deepcopy(scaled_train_test)

        # --------------------------- #
        # Iterate over the Variables  #
        # --------------------------- # 
        for variable, var_dict in permuted_train_test.items():

            if variable not in ['UTME', 'UTMN', 'Y']:
                
                # ----------------------- #
                # Iterate over the Sites  #
                # ----------------------- #
                
                for site_name, site_dict in var_dict.items():
                    
                    # ----------------------- #
                    # Iterate over the Images #
                    # ----------------------- #
                    
                    for tr_or_test in site_dict.keys():
                        
                        for img_idx in np.arange(
                            permuted_train_test[variable][site_name][tr_or_test].shape[0]
                        ):
                            # ----------------------- #
                            # Permute the Feature     #
                            # ----------------------- #
                            np.apply_along_axis(
                                np.random.shuffle
                                ,axis=-1
                                ,arr=permuted_train_test[variable][site_name][tr_or_test][img_idx,:,:,f_idx]
                            )
        model_assessment = model_assessor(
            X=scaled_train_test['X'][site]
            ,Y=scaled_train_test['Y'][site]
            ,loaded_model=model
            ,val_index=val_index
        )

        model_assessment_per = model_assessor(
            X=permuted_train_test['X'][site]
            ,Y=permuted_train_test['Y'][site]
            ,loaded_model=model
            ,val_index=val_index
        )
        
        output.loc[feat] = pd.DataFrame.subtract(
            model_assessment_per, model_assessment
        ).loc['test'].to_list()
        
    return output
