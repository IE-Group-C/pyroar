# -*- coding: utf-8 -*-
from manipulator import Manipulator
import numpy as np
from sklearn.impute import SimpleImputer

class Imputer(Manipulator):
     def __init__(self,df,metadataO):
       metadata=metadataO.copy()
       metadata = super().metadata_verifier(df,metadata)
       tmp_metadata=self.remove_no_actions_per_type(metadata,'impute')
       super().__init__(df,metadata)
       keys = metadata.keys()
       
       # initialization class variables
     
       self.df=df
       self.fitted_imputer={}
       self.imputer_dict = {x:tmp_metadata[x]['impute'] for x in keys}
       self.is_fit = False
      
               
         
     def fit(self):
            '''
             This function fits the data for each imputer type of the allowed imputers in this package. 
             The allowed imputers are ['ConstantImputer','MostFrequentImputer','MeanImputer','MedianImputer']
             
             Manages
             -------
             is_fit : bool
                 This flag confirms if all the fits were completed successfuly or there was an issue. 
    
            '''        
        # fit all imputer
            try:
                self.is_fit=True # to indicate if fit function completed
                # for transformation, for constant imputer : 'ConstantImputer_20'
                avaliable_transformations=['ConstantImputer','MostFrequentImputer','MeanImputer','MedianImputer']
                # check transformation availability
                if super().check_transformation(avaliable_transformations,self.imputer_dict.values(),'imputation')==True:
                    # check columns availability 
                   if  super().check_columns(self.df.columns,self.imputer_dict.keys()) ==True:
                       
                       for x, y in self.imputer_dict.items():
                           # check imputer type and fit
                           if y.split('_')[0] == 'ConstantImputer': 
                              
                               if super().is_float(y.split('_')[1])==True and super().check_type(self.df[[x]],'Numerical')==True:  
                                   imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=int(y.split('_')[1]))
                                   self.fitted_imputer[x]= imputer.fit(self.df[[x]])
                               
                               elif super().is_float(y.split('_')[1])!=True and super().check_type(self.df[[x]],'Categorical')==True: 
                                   imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=y.split('_')[1])
                                   self.fitted_imputer[x]= imputer.fit(self.df[[x]])
    
                               else:
                                   self.is_fit=False
                                   print('The constant value is not matching  ',x,'column type')
                                   
                                   
                           elif y == 'MostFrequentImputer':
                               imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                               self.fitted_imputer[x]= imputer.fit(self.df[[x]])
                               
                           elif y == 'MeanImputer':
                               # check data type
                               if super().check_type(self.df[x],'Numerical')==True: # only for 'Mean Imputer','Median Imputer'
                                   imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                                   self.fitted_imputer[x]= imputer.fit(self.df[[x]])
                                   
                               else:
                                   self.is_fit=False
                                   print('There was an issue in Mean Imputer')
                                   
                           else:
                               # check data type
                               if super().check_type(self.df[x],'Numerical')==True: # only for 'Mean Imputer','Median Imputer'
                                   imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                                   self.fitted_imputer[x]= imputer.fit(self.df[[x]])
                                   
                               else:
                                   self.is_fit=False
                                   print('There was an issue in Median Imputer Imputer')

                                   
                   else:
                        self.is_fit=False
                        
                else:
                   self.is_fit=False
                   print("This imputation type is not allowed")
                   
                #return Imputer_flag
            except:
                print('an error occurs while fitting proposed imputers, please ensure the fitted data are compatible with the available imputers in this package')
                self.is_fit=False
                #return Imputer_flag
            
            
            
            
            
     # Transform all imputer       
     def transform(self,keep_original=False):
         '''
         This function transfomrs the data by performing the recomended imputation approach
         
         Parameters
         ----------
         keep_original : bool Default=False
             This variable decides if the orignial column should be kept or deleted. The original column is deleted by default.  

         Returns
         -------
         df:    DataFrame
             It returns the updated and imputed dataframe

         '''
         # loop through fitted scaler
         if(self.is_fit):
             try:
                 for t,y in self.fitted_imputer.items():
                     if(keep_original):
                         self.df[t+"_imputed"] = y.transform(self.df[[t]])
                     else:
                         self.df[t] = y.transform(self.df[[t]])
                     
                 return self.df
             except:
                  print('an error occurs while performing imputation, please refit the data and apply imputers again')
         else:
             print("You need to fit the imputer first")