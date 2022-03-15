# -*- coding: utf-8 -*-
from transformation import transformation
from encoding import encoding
from imputation import imputation
from scaling import scaling
from manipulator import Manipulator
import pandas as pd

class General:
    def __init__(self,df,metadata):
       self.df=df
       self.metadata=metadata
       self.is_fit = False
       self.manipulator = Manipulator(df,metadata)  
       self.transformer = transformation.Transformer(df,metadata)
       self.encoder = encoding.Encoder(df,metadata)
       self.imputer = imputation.Imputer(df,metadata)
       self.scaler = scaling.Scaler(df,metadata)
       
       
       
    # def fit(self,target):
    # #add check if transformation
    #     try:            
    #        self.encoder=encoding.Encoder(self.df,self.metadata)
    #        self.encoder.fit(target)
    #        if self.encoder.is_fit==True:
    #            # encoder.transform()
    #            # print(' encoding operations were completed successfully')
    #            pass
    #        else:
    #            print('an error occured while fitting encoder')       
           
    #        self.imputer=imputation.Imputer(self.df,self.metadata)
    #        self.imputer.fit()
    #        if self.imputer.is_fit == True:
    #            # imputer.transform()
    #             # print(' imputation operations were completed successfully')
    #             pass
    #        else:
    #            print('an error occured while fitting imputers')
               
               
    #        self.scaler=scaling.Scaler(self.df,self.metadata)
    #        self.scaler.fit()
    #        if self.scaler.is_fit==True:
    #            # scaler.transform()
    #            # print(' imputation operations were completed successfully')
    #            pass
    #        else:
    #            return 'an error occured while fitting scalers'
           
    #        self.is_fit = True
    #        print("fit was completed successfuly")
    #     except Exception as e:
    #         print("The fit wasn't completed successfully. An Error has occured")
    #         print(e)
            
 
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
    #add check if transformation
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

            
              