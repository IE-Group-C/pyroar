
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels import robust
from scipy import stats

from transformation import transformation
from encoding import encoding
from imputation import imputation
from scaling import scaling
from manipulator import Manipulator

class Pyroar:
    
    def __init__(self ):
        pass
        self.df = None
        self.metadata= {}
        self.is_fit = False
        
        self.manipulator = None
        self.transformer = None
        self.encoder = None
        self.imputer = None
        self.scaler = None
        
        
    # End of __init__
        
    """
    ============================
    DF Exploratory Data Analysis
    ============================
    Easy and quick way to get the most important statistics for
    a dataframe with a single command showing several metrics.
    """
    def scan(self, df, target, dropThreshold=0.5, catsLblEncodeThreshold=20, catsDropThreshold=100, outlier_scale_lim = 1.5, outlier_scale_frac=0.005, outlier_drop_lim=3.0, pvalue=0.05):
        """
        
        Applies a number of checks over the dataframe and returns different statistics and recommendations
        for a number of transformations for different columns in the dataframe.

        Parameters
        ----------
        df : string, dataframe name 
            Python pandas dataframe.
        target : string, target column name in dataframe
            Binary classification target column name in the dataframe (column must exist in the dataframe)
        dropThreshold : float, optional
            Threshold of missing values percentage over which to recommend dropping a column from the dataframe.
            The default is 0.3.
        catsLblEncodeThreshold : integer, optional
            Threshold of catagorical variables catagories count over which to recommend applying label encoding.
            Must be lower than the value in 'catsDropThreshold'.
            Catagorical variables lowwer than the threshold will be recommended to be one-hot encoded.
            The default is 20.
        catsDropThreashold : integer, optional
            Threshold of catagorical variables catagories count over which to recommend dropping a column from the dataframe.
            The default is 100.
        outlier_scale_lim : float, optional
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            The identified outliers will be recommended to be treated by applying robust scaler
            or will be ignored depending on the fraction specified in 'outlier_scale_frac'
            The default is 1.5.
        outlier_scale_frac : float, optional
            Fraction of outliers over which to recommend applying robust scaler based on the limit specified in 'outlier_scale_lim'
            If fraction of outliers is below this fraction, outliers will not be treated (ignored).
            The default is 0.005.
        outlier_drop_lim : float, optional
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            Those outliers will be recommended to be dropped.
            The default is 3.0.
        pvalue : float, optional
            Probability value for null-hypothesis significance testing used in statistical tests used in this function.
            The default is 0.05.

        Returns
        -------
        dictionary
            Dictionary of all recommendations of the function for each column to be applied in other functions.

        """
        
        self.num_features = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_features = df.select_dtypes(include=np.object).columns.tolist()
        
        
        self.df = df
        dropThreshold = dropThreshold
        catsLblEncodeThreshold = catsLblEncodeThreshold
        catsDropThreshold = catsDropThreshold
        outlier_scale_lim = outlier_scale_lim
        outlier_scale_frac = outlier_scale_frac
        outlier_drop_lim = outlier_drop_lim
        pvalue = pvalue
        
        for col in df.columns:
            self.metadata.update({col: {}})
            
        print("Head of dataframe:\n")
        print(df.head())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nTail of dataframe:\n")
        print(df.tail())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nNames of existing columns:\n")
        print(df.columns)
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nData types of the existing columns:\n")
        print(df.info())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nNumbers of columns with same data type:\n")
        print(df.dtypes.value_counts())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nMemory usage of each column:\n")
        print(df.memory_usage())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nShape of the dataframe (rows/columns):\n")
        rows = df.shape[0]
        cols = df.shape[1]
        print("There are " + str(rows) + " rows and " + str(cols) + " columns in this dataframe.")
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        
        print("\nNumber of occuring duplicates in the dataframe:\n")
        dupRows = df.duplicated().sum()
        print("There are " + str(dupRows) + " duplicated rows in the dataframe.")
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nGernalized Auto Plots:\n")
        self.plot()
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\Correlation Plot:\n")
        self.correlation_plot()
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nCheck the Missing Values per column:\n")
        self.scan_missing_values(dropThreshold)
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        

        #### Note: This section should be applied to categorical Variables ONLY ************
        print("\nCheck Unique Classes per Categorical Features")    
        #self.scan_catorical_classes(catsDropThreshold, catsLblEncodeThreshold )
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
               
        ######################  IMPLEMENT OUTLIER DETECTION  ####################
        #### Note: This section should be applied to numerical Variables ONLY ************
        print("\nCheck Unique Classes per Categorical Features")  
        self.scan_outliers( outlier_scale_lim, outlier_scale_frac, outlier_drop_lim)
        outlier_scale_lim, outlier_scale_frac, outlier_drop_lim
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        print("\nStandard statistics for each column:\n")
        print(df.describe())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
           
        print("\nCorrelations between columns:\n")
        print(df.corr())
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
                       
        for var in self.num_features:
            print(df[var])
            self.df[var] = pd.to_numeric(self.df[var], errors='coerce')
            self.df[var] = self.df[var].fillna(0)
        
        ########################### STATS #######################    
        #### Statistical Testing for Numerical Features 
        print("\nStatistical Test for Numerical Features using Shapiro and arque-Bera")
        self.stat_test_numerical( pvalue)
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        #### Statistical Testing for Categorical Features 
        print("\nStatistical Tests for Categorical Features using Chai-square")
        self.stat_test_categorical( pvalue, target)
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        
        return self.metadata
         
        
    # End of scan()
    
    
    def prep(self):
        """
        This funcgtion handles the missing values, outliers, and 
        categorical variables with a large nubmer of unique values

        Parameters
        ----------
        None.
        Returns
        -------
        None.
        """
        # cleaning
        print("\n\n ==== Preperation function started =====")
        for col in self.metadata:
           print('\nColumn Name:', col ,  ' ** values:', self.metadata[col])
           # implement your fit methods here
           
           trans = self.metadata[col]
           
           # Treating Missing Values by Dropping
           # {'Drop': 'DropColumn'}   ==> Too many missing values
           key = 'Drop'
           if key in trans:
                print(trans[key])
                if trans[key] == 'DropColumn':
                    if col in self.df.columns:
                        self.df.drop([col], axis=1, inplace=True)
           
            # {'Encoding': 'DropColumn'} ==> too many class for cat variables
           key = 'MissingValues'
           if key in trans:
                if trans[key] == 'DropColumn':
                    if col in self.df.columns:
                        self.df.drop([col], axis=1, inplace=True)
                    
            
           # {'DropExtermeOutliersValues': outlier_drop_lim} ==> Drop Extreme outliers
           key = 'DropExtermeOutliersValues'
           if key in trans:
                if trans[key][0] < 1.5 :
                    raise Exception("Sorry, You cannot set limit below 1.5 IQR")
                    
                elif col in self.df.columns:
                    # Calculate Q3 and Q1
                    # calculate the IQR
                    print(self.df[col], trans[key])
                    
                    # # find outliers extreme_upper_bound
                    extreme_upper_bound = trans[key][1]
                    # # find outliers extreme_lower_bound
                    extreme_lower_bound = trans[key][2]
                    
                    # drop rows with values <lower_bound or values >upper_bound
                    #self.df = self.df[self.df[col] != np.nan and ( self.df[col] > extreme_upper_bound or self.df[col] < extreme_lower_bound)  ]
                    self.df = self.df[ (self.df[col] > extreme_upper_bound) | (self.df[col] < extreme_lower_bound) ]
                    
    # End of prep()      
    
    
    def fit(self,X,y):
        """
        Applies fit methods for a number of recommended transformations given in the 'scan' function
        for different columns in the dataframe.        

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used for later transformation (e.g for scaling or categories of each feature for one-hot encoder)
        y : series-like of shape (n_samples,)
            Binary target values (class labels) as integers used for Weight of Evidence (WoE) transformation.
        
        """
        
        self.manipulator = Manipulator(self.df,self.metadata)  
        self.transformer = transformation.Transformer(self.df,self.metadata)
        self.encoder = encoding.Encoder(self.df,self.metadata)
        self.imputer = imputation.Imputer(self.df,self.metadata)
        self.scaler = scaling.Scaler(self.df,self.metadata)
    
        try:
            if isinstance(X, pd.DataFrame):
                try:            
                   self.encoder=encoding.Encoder(X,self.metadata)
                   self.encoder.fit(y)
                   if self.encoder.is_fit==True:
                       # encoder.transform()
                       # print(' encoding operations were completed successfully')
                       pass
                   else:
                       print('an error occured while fitting encoder')       
                   
                   self.imputer=imputation.Imputer(X,self.metadata)
                   self.imputer.fit()
                   if self.imputer.is_fit == True:
                       # imputer.transform()
                        # print(' imputation operations were completed successfully')
                        pass
                   else:
                       print('an error occured while fitting imputers')
                       
                    
                   self.scaler=scaling.Scaler(X,self.metadata)
                   self.scaler.fit()
                   if self.scaler.is_fit==True:
                       # scaler.transform()
                       # print(' imputation operations were completed successfully')
                       pass
                   else:
                       return 'an error occured while fitting scalers'
                   
                   self.is_fit = True
                   print("fit was completed successfuly")
                except Exception as e:
                    print("The fit wasn't completed successfully. An Error has occured")
                    print(e)
            else:
                raise Exception("X needs to be of type DataFrame")
        except Exception as e:
            print(e)
            
    def transform(self,keep_original=False):
        """   

         Perform transformations for the fitted transformers given in the 'fit' function
         for different columns in the dataframe.        

         Parameters
         ----------
         keep_original: bool Default: False
                This bool variable deiced to either keep the transformed values in seperate columns or replace the exisitng column
                

         Returns
         -------         
         The funciton doesn't reaturn anything though it updates the self.df value for this object. The result can be accessed using object.df

        """
        try:
            if(self.is_fit):
                
                try:
                    self.df = self.imputer.transform(keep_original)
                except Exception as e:
                    print("There was an error in the Imputer class")
                    print(e)
                    
                    
                try:
                   self.transformer=transformation.Transformer(self.df,self.metadata)
                   self.df = self.transformer.transform(keep_original)     
                except Exception as e:
                    print("There was an error in the Transformer class")
                    print(e)
        
        
                try:
                    self.df = self.scaler.transform(keep_original)
                except Exception as e:
                    print("There was an error in the Scaler class")
                    print(e)
        
        
                try:
                    self.df = self.encoder.transform(keep_original)
                except Exception as e:
                    print("There was an error in the Encoder class")
                    print(e)
                   
                print("Transform was completed successfully")
            else:
                raise Exception ("You can't transform before you fit. Please run the fit() function first")
            
            
        except Exception as e:
            print(e)
            
            
    def get_recommendad_feature_engineering(self,df_format=False):
        '''
         This function is intended to print the metadata in dataframe format to make it easier for user to see the feature engineering that will 
         be done to the data.
         
        Parameters
        ----------
        df_format : bool, optional
            it checks if we want to return the recommeneded the feature engineering in DataFrame format or not. The default is False.

        Returns
        -------
        df or dict
            Recommended feature engineering in DataFrame format or in dict format.
        '''
        try:
            if(df_format):
                tmp_df = self.manipulator.get_recommendad_feature_engineering(metadata = self.metadata)
                print("Here are the recommended feature engineering for the fit and transform")
                print(tmp_df)
                return tmp_df
                
            else:
                print("Here are the recommended feature engineering for the fit and transform")
                print(self.metadata)
                return self.metadata
        except Exception as e:
            print("There was an issue in getting recommended feature engineering")
            print(e)
        
    
    
    def modify_recommended_feature_engineering(self,modified_metadata,metadata=None,inplace=False):
        '''
        This function is used to allow the user to modify the recommended feature engineerings that were given by the scan funciton. 
        Using this function the user can pass a dictionary that will modify only the parts specified in the modified dictionary and use the modified types if allowed 
        in the following fit and transform calls. 
        
        This funciton will could overwrite the metadata based on the inplace flag and return an updated metadata

        Parameters
        ----------
        modified_metadata : dict
            his is the dictionary containing the metadata that the user would like to change in the metadata.
            
            The expected format is as follows:
                {ColumnName:{TransformationType:Tranformation}}
         e.g.
                {'Gender':{'scaling':'RobustScaler'}} 
                
        metadata : dict, optional
            This is the metadata that would be changed. The deafult is to modify the metadata stored already in the object.  The default is None.
            
        inplace : bool, optional
            This bool is used to either replace the metadata stored in the object or just return the new metadata to the user. The default is False.

        Returns
        -------
        None.

        '''
        
        
        try:
            if(inplace):
                self.metadata = self.manipulator.modify_recommended_feature_engineering(modified_metadata,metadata)
            else:
                self.manipulator.modify_recommended_feature_engineering(modified_metadata,metadata)
        except Exception as e:
            print("There was an issue in modifying the recommended feature engineering metadata")
            print(e)

    
    def count_outlier_numircal(self, x, limLow, limHigh):
        """
        
        Counts the number of outliers in numerical columns in the dataframe
        
        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data in the numerical columns.
        limLow : float
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            The identified outliers to be treated later by applying robust scaler
            or will be ignored depending on the scan fraction as specified in 'outlier_scale_frac'
        limHigh : float
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            Those outliers will be dropped later.
        
        Returns
        -------
        outliers_normal : integer
            count of normal outliers to be treated later by applying robust scaler or to be ignored depending on the scan fraction as specified in 'outlier_scale_frac'.
        extreme_outliers : integer
            count of extreme outliers to be removed later.

        """
        
        #Q1 = np.percentile(self.df[x], 25, interpolation = 'midpoint')
        Q1 = np.quantile(self.df[x], q=0.25)
        #Q3 = np.percentile(self.df[x], 75, interpolation = 'midpoint')
        Q3 = np.quantile(self.df[x], q=0.75)
        IQR = Q3 - Q1
            
        #Old_length=len(x)
        #print("Old length: ", Old_length)
        
        # Upper bound for Exetreme outliers
        upper_extreme = len(np.where(self.df[x] >= (Q3+limHigh*IQR)))
        # Lower bound for Exetreme outliers
        lower_extreme = len(np.where(self.df[x] <= (Q1-limHigh*IQR)))
        extreme_outliers = upper_extreme + lower_extreme
        
        
        # Upper bound for Exetreme outliers
        upper = len(np.where(self.df[x] >= (Q3+limLow*IQR)))
        # Lower bound for Exetreme outliers
        lower = len(np.where(self.df[x] <= (Q1-limLow*IQR)))
        outliers_normal =  upper + lower - extreme_outliers
        
        print(Q1, Q3, IQR, outliers_normal, extreme_outliers, Q1-limHigh*IQR, Q3+limHigh*IQR)
        return outliers_normal, extreme_outliers, Q1-limHigh*IQR, Q3+limHigh*IQR
    # End of count_outlier_numircal()
    
    def detect_outliers_zscore(self, data):
        """
        
        Detects outliers for features with normal distribution. Standard zscore method is used to determine the outliers.
        If we have a normal distribution we use a standard zscore.

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data in the numerical columns.

        Returns
        -------
        outliers : integer
            count of outliers identified using zscore method.

        """
        # threshhold for the outliers detection limit is taken to be equal to 3
        outliers =[]
        threshold = 3
        mean = np.mean(data)
        std = np.std(data)
        
        for x in data:
            zscore = (x-mean)/std
            if np.abs(zscore)> threshold:
                outliers.append(x)
        return outliers
    # End of detect_outliers_zscore()
    
    
    
    def detect_outliers_Mzscore(self, data):
        """
        
        Detects outliers for features with  non-normal distribution. Modified zscore method is used to determine the outliers.

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data in the numerical columns.

        Returns
        -------
        outliers : integer
            count of outliers identified using zscore method.

        """
        # threshhold for the outliers detection limit is taken to be equal to 3
        outliers =[]
        threshold = 3
        # compute modified z
        Med = np.median(data)
        MAD = robust.mad(data)
        
        for x in data:
            Mzscore = stats.norm.ppf(.75)*(x-Med) / MAD
            if np.abs(Mzscore)> threshold:
                outliers.append(x)
        return outliers
    # End of detect_outliers_Mzscore()
    
    
    def plot(self):
        """
        
        Makes plots of a dataframe.

        Parameters
        ----------
        None.

        Returns
        -------
        Plots of the dataframe

        """
        numeric=self.df.select_dtypes(include=['float64'])
        plt.plot(numeric)
        #plt.display()
    # End of plot()
    
    def correlation_plot(self, triangle = True, shrink = 0.5, orientation = "vertical", my_palette = None):
        """
        
        Obtains correlation matrix and associated correlation plot

        Parameters
        ----------
        data : dataframe
            Data set to calculate correlations from
        -triangle : boolian, optional
            Either True or False; default is True, Whether to plot the full heatmap of correlations or only the lower triangle.
        shrink : float, optional
            Governs the size of the color bar legend. See matplotlib.pyplot.colorbar() for more information.
        orientation : string, optional
            Either "vertical" or "horizontal"; default is "vertical".
            Governs where the color bar legend is plotted. See matplotlib.pyplot.colorbar() for more information.
        my_palette : colormap name or object, or list of colors, optional
            The mapping from data values to color space. If not provided, the default will depend on center of set.

        Returns
        -------
        corr_mat : tuple of (corr_mat, ax.figure)
            corr_mat is a pandas.DataFrame() holding the correlations.
            ax.figure is the plot-object of the heatmap.

        """
        corr_mat = self.df.corr(method="pearson")
    
        if triangle == True:
          mask = np.zeros_like(corr_mat)
          mask[np.triu_indices_from(mask)] = True
        else:
          mask = None
    
        with sns.axes_style("white"):
            plt.figure(figsize=(15,15))
            ax = sns.heatmap(corr_mat, vmin = -1,vmax= 1, square=True, cmap = my_palette, cbar_kws={"shrink": shrink, "orientation": orientation}, mask=mask)
            ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 6)
            ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 6)
        return corr_mat, ax.figure
    # End of correlation_plot()
        
    
    def scan_missing_values(self, dropThreshold):
        """
        
        Scans for missing values in the dataframe and stores recommendation for the user in a dictoinary.
        Called by the 'scan' function.
        
        Parameters
        ----------            
        dropThreshold : float
            Threshold of missing values percentage over which to recommend dropping a column from the dataframe.

        Returns
        -------
        None.

        """
        print("\nNumber of occuring NULL/NA values per column:\n")
        print(self.df.isnull().sum())
        print("\nPercentage of occuring NULL/NA values per column:\n")
        missing_perentage = self.df.isnull().sum()/self.df.isnull().count() * 100
        print( missing_perentage)
        for col, val in zip(self.df.columns , missing_perentage):
            if(val >= dropThreshold):
                 # drop the column
                 print('Column: {} , metadata: Drop Column.  Too many missing values {}%'.format(col, val))
                 self.metadata[col].update(  {'MissingValues': 'DropColumn'})
            elif val>0:
                 print('Column: {} , metadata: Imputation for Missing Values is required.  Number of missing values {}%'.format(col, val))
                 self.metadata[col].update(  {'Imputation': 'Mean'})
            else: # zero missing values
                 print('Column: {} , metadata: No imputation is required.  Number of missing values {}%'.format(col, val))
                 #self.metadata[col].update(  {'Imputation', 'Mean'})
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
    
    
    def scan_catorical_classes(self, catsDropThreshold,catsLblEncodeThreshold ):
        """
        
        Scans for catagorical columns in the dataframe and stores recommendation on how to handle them for the user in a dictoinary.
        Called by the 'scan' function.        

        Parameters
        ----------
        catsLblEncodeThreshold : integer
            Threshold of catagorical variables catagories count over which to recommend applying label encoding.
            Must be lower than the value in 'catsDropThreshold'.
            Catagorical variables lowwer than the threshold will be recommended to be one-hot encoded.
        catsDropThreashold : integer
            Threshold of catagorical variables catagories count over which to recommend dropping a column from the dataframe.

        Returns
        -------
        None.

        """
        print("\nNumber of occuring unique values per column:\n")
        n_unique = self.df.nunique()
        print(""+n_unique)
        for col, val in zip(self.df.columns , n_unique):
             if(val > catsDropThreshold):
                 # drop the column
                 print('Column: {} , metadata: Drop.  Too many unique values {}'.format(col, val))
                 self.metadata[col].update(  {'Encoding': 'DropColumn'})
             elif val >catsLblEncodeThreshold:
                 # label encoder
                 print('Column: {} , metadata: Use FrequancyEncoder.  Moderate number of unique values {}'.format(col, val))
                 self.metadata[col].update(  {'Encoding': 'FrequancyEncoder'})
             else:
                 #one hot encoding
                 print('Column: {} , metadata: Use OneHotEncoder.  Limitted number of unique values {}'.format(col, val))
                 self.metadata[col].update(  {'Encoding': 'OneHotEncoder'})
                 
    def scan_outliers(self, outlier_scale_lim, outlier_scale_frac, outlier_drop_lim):
        """
        
        Scans for outliers in the dataframe and stores recommendation on how to handle them for the user in a dictoinary.
        Called by the 'scan' function.

        Parameters
        ----------
        outlier_scale_lim : float
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            The identified outliers will be recommended to be treated by applying robust scaler
            or will be ignored depending on the fraction specified in 'outlier_scale_frac'
        outlier_scale_frac : float
            Fraction of outliers over which to recommend applying robust scaler based on the limit specified in 'outlier_scale_lim'
            If fraction of outliers is below this fraction, outliers will not be treated (ignored).
        outlier_drop_lim : float
            Number of Interquartile Range statistical measure (IQR) over which outliers are identified using Interquartile Range statistical measure (IQR).
            Those outliers will be recommended to be dropped.

        Returns
        -------
        None.

        """
        num_out_low_lim = {}
        num_out_high_lim = {}
        num_out_lower_bound = {}
        num_out_upper_bound = {}
        for col in self.num_features:
           low, high, lower_bound, upper_bound =  self.count_outlier_numircal(col, outlier_scale_lim, outlier_drop_lim)
           num_out_low_lim[col] = int(low)
           num_out_high_lim[col] = int(high)
           num_out_lower_bound[col] = lower_bound
           num_out_upper_bound[col] = upper_bound
        print(num_out_low_lim)
        
        #metadata['OutlierValuesDropLimit'].update(outlier_drop_lim )
        ## Drop the outlier values that are greater than upper limit (e.g. 3 IQR )
        print("print the dictionarty 547")
        print(self.metadata)
        for col, val in num_out_high_lim.items():
             if int(val) > 0 :  
                 print('Column: {} , metadata: Drop Outlier Value.  There are {} outlier values that are more than {} IQR'.format(col, val, outlier_drop_lim))
                 
                 self.metadata[col].update(  {'DropExtermeOutliersValues': [outlier_drop_lim, num_out_lower_bound[col], num_out_upper_bound[col] ]})
                 #{ 'Age': { 'DropExtermeOutliersValues': 5 , 'scaling': 'StandardScaler' },          }
        ## use robustScalaer if the column has 3% or more of outliers that are more
        ## the lower outliers limit (e.g. 1.5 IQR)
        for col, val in  num_out_low_lim.items():
             if( (val/ len(self.df)) > outlier_scale_frac ):  # more than e.g 0.1% are outliers ==> Robust Scaler
                 print('Column: {} , metadata: Use RobustScaler.  There are {:0.2f} % outlier values that are more than {} IQR'.format(col, (val/len(self.df)) , outlier_scale_lim))
                 self.metadata[col].update(  {'Scaling': 'RobustScaler'})
                 
                 
    def stat_test_numerical(self, pvalue):
        """
        
        Tests for normality using Jarque-Bera or Shapiro tests on numerical variables in the dataframe
        and stores recommendation on how to scale them for the user in a dictionary.
        Also checks normality by testing reciprocal (1/x), logarithmic (log x), and square exponent (x^2) functions transformations on numerical variables.
        To perform the Jarque-Bera goodness, first of all, it is needed to check that the sample is above 2000 (n>2000) data points.
        Otherwise it is recommended to use the Shapiro-Wilk tests to check for normal distribution of the sample.
        Called by the 'scan' function.

        Parameters
        ----------
        pvalue : float
            Probability value for null-hypothesis significance testing used in statistical tests used in this function.

        Returns
        -------
        None.

        """
        for var in self.num_features:
            print(self.df[var])
            self.df[var] = pd.to_numeric(self.df[var], errors='coerce')
            self.df[var] = self.df[var].fillna(0)
            
        
        
        # To perform the Jarque-Bera goodness, first of all, it is needed to check that the sample is above 2000 (n>2000) data points.
        # Otherwise it is recommended to use the Shapiro-Wilk tests to check for normal distribution of the sample.
        
        
            ### test for normality Shapiro and Jarqu-Bera)
            for col in self.num_features:
                      
                
                if len(self.df) <=2000: # use shapiro
                    statistic, pval = stats.shapiro(self.df[col])
                else:   # use jarque_bera
                    statistic, pval = stats.jarque_bera(self.df[col])
                
                if pval > pvalue:
                    print("Dataset df has a normal distribution on var {}, as the null hyposthesis was not rejected for having a {:.4f} p-Value".
                         format(col, pval))
                    self.metadata[col].update(  {'Scaling': 'StandardScaler'})
                else:
                    print("Dataset df does not have a normal distribution on var {}, as the pvalue is {:.4f}".
                          format(col,pval))
                    print("Try different transformations on var {}".format(col))
                    #
                    ### test transformations   1/X    log x    x^2
                    # 1/X transofrmation
                    #X = pd.map(lambda i: 1/i, df[col])
                    x = 1/self.df[col]
                    if len(self.df) <=2000: # use shapiro
                        statistic, pval = stats.shapiro(x)
                    else:   # use jarque_bera
                        statistic, pval = stats.jarque_bera(x)
                    if pval > pvalue:
                        print("Dataset df has a normal distribution on var 1 / {}, as the null hyposthesis was not rejected for having a {:.4f} p-Value".
                         format(col, pval))
                        
                        print("Recommendation: Transfor feature to 1/x")
                        self.metadata[col].update(  {'Transformation': 'reciprocal'})
                        self.metadata[col].update(  {'Scaling': 'StandardScaler'})
                    else:
                        # print("Dataset df does not have a normal distribution on var {}, as the pvalue is {:.4f}".
                        #       format(col,pval))
                        # Check log x transformation
                       
                        x = np.log(self.df[col].replace(0,1))
                        if len(self.df) <=2000: # use shapiro
                            statistic, pval = stats.shapiro(x)
                        else:   # use jarque_bera
                            statistic, pval = stats.jarque_bera(x)
                        if pval > pvalue:
                            print("Dataset df has a normal distribution on var log({}), as the null hyposthesis was not rejected for having a {:.4f} p-Value".
                             format(col, pval))
                            
                            print("Recommendation: Transfor feature to 1/x")
                            self.metadata[col].update(  {'Transformation': 'log10'})
                            self.metadata[col].update(  {'Transformation': 'StandardScaler'})
                        else:
                            ##  Chekc X^2 transofrmation
                            x = self.df[col]**2
                            if len(self.df) <=2000: # use shapiro
                                statistic, pval = stats.shapiro(x)
                            else:   # use jarque_bera
                                statistic, pval = stats.jarque_bera(x)
                            if pval > pvalue:
                                print("Dataset df has a normal distribution on var {}**2, as the null hyposthesis was not rejected for having a {:.4f} p-Value".
                                 format(col, pval))
                                
                                print("Recommendation: Transfor feature to 1/x")
                                self.metadata[col].update(  {'Transformation': 'squared'})
                                self.metadata[col].update(  {'Scaling': 'StandardScaler'})
                            else:
                                # none of the transformations worked ==> Appy MinMaxScaler
                                self.metadata[col].update(  {'Scaling': 'MinMaxScaler'})
    # End of stat_test_numerical()
      
    def stat_test_categorical(self, pvalue, target):
        """
        
        Implements Chi Square test for categorical nominal variables in the dataframe
        and stores recommendation whether to keep them or remove them for the user in a dictionary.
        Called by the 'scan' function.

        Parameters
        ----------
        pvalue : float
            Probability value for null-hypothesis significance testing used in statistical tests used in this function.
        target : string, target column name in dataframe
            Binary classification target column name in the dataframe (column must exist in the dataframe)

        Returns
        -------
        None.

        """
        ########################### Stat_3. IMPLEMENT ChaiSquare-Test for categorical nominal variables
        ### PROBLEM: we do not have the population to perform Chai-Square test
        # Source: https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        
        for col in self.cat_features:
            self.df[col] = label_encoder.fit_transform(self.df[col])


        from sklearn.feature_selection import chi2
        print("chi squared test here",self.cat_features)
        X = self.df[self.cat_features]
        print(X.head())
        y = self.df[target] # we have to ask the user to specify the target variable at the beginning and then make it binary 1,0

        chi_scores = chi2(X,y)
        
        p_values = pd.Series(chi_scores[1],index = X.columns)
        p_values.sort_values(ascending = False , inplace = True)
        #p_values<0.05
        i = 0
        for pval in p_values:
            col = p_values.index[i]
            if pval >= pvalue:   
                print( " The variable {} is recommended to be removed since it is independent from the target variable. We fail to reject the null hypothesis as p-value >= 5%: {} ".format(col,pval))
                self.metadata[col].update(  {'Drop': 'DropColumn'}) #the recommendation is to drop the column. Also the 5% shouldn't be hard-coded
            else:
                print( " The variable {} is recommended to be kept since it is dependent on the target variable. We reject the null hypothesis as p-value < 5%: {} ".format(col,pval))
            i = i+1
    # End of stat_test_categorical()           
            
    def printDf(self):
         print(self.df)
         print(self.metadata)
        
   # End of printDf()
    
    def getDf(self):
         return self.df
         
        
   # End of printDf()
    
    
   