'''

This utilities script is intended for the students attending the Early Career Scientist Machine Learning Workshop at the American Geophysical Union Conference 2019. 

Joel A. Gongora
Data Scientist
PhD Candidate
Boise State University

email:  geodatamex@gmail.com
        joel.gongora@elderresearch.com
'''


import gzip
import pickle
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
import pandas as pd

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
