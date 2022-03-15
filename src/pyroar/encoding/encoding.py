# -*- coding: utf-8 -*-

from manipulator import Manipulator
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import sklearn

'''
Special credit to the category_encoders library that we used. 
'''


class Encoder(Manipulator):
     def __init__(self,df,metadataO):
       metadata=metadataO.copy()
       metadata = super().metadata_verifier(df,metadata)
       super().__init__(df,metadata)
       keys = metadata.keys()
       tmp_metadata=self.remove_no_actions_per_type(metadata,'encoding')
       # initialization class variables
       
       self.df=df
       self.fitted_encoder={}
       self.encoder_dict = {x:tmp_metadata[x]['encoding'] for x in keys}
       self.is_fit = False
       
       
     # fit all Encoder
     def fit(self,y):
            '''
             This function fits the data for each encoder type of the allowed encoders in this package. 
             The allowed encoders are ['OHE','FREQ','WOE']
             
             Parameters
             ----------
             y : array of bools
                 y is the target that we need to check against. This is expected to be an array of bool 
    
             Manages
             -------
             is_fit : bool
                 This flag confirms if all the fits were completed successfuly or there was an issue. 
    
            '''               
            try:
                self.is_fit=True # to indicate if fit function is completed successfully 
               
                avaliable_transformations=['OHE','FREQ','WOE']
                # check transformation availability
                if super().check_transformation(avaliable_transformations,self.encoder_dict.values(),'encoding')==True:
                   # check columns availability 
                   if  super().check_columns(self.df.columns,self.encoder_dict.keys()) ==True:
                       for x, t in self.encoder_dict.items():
                           # check imputer type and fit
                           if t == 'OHE':
                               
                               encoder = OneHotEncoder(handle_unknown='ignore')
                               self.fitted_encoder[x]= encoder.fit(self.df[[x]])
                                
                           elif t == 'FREQ':
                               try:
                                   encoder = ce.CountEncoder(cols=[x])
                                   self.fitted_encoder[x]= encoder.fit(self.df[[x]])
                               except ValueError as e:
                                   print("Frequency encoder")
                                   print(e)
                                   self.is_fit=False
                               except KeyError as ke:
                                   print("Frequency encoder")
                                   print(ke)
                                   self.is_fit=False
                             
                           elif t == 'WOE':
                               try:
                                   encoder = ce.WOEEncoder(cols=[x])
                                   self.fitted_encoder[x]= encoder.fit(self.df[[x]],y)
                               except ValueError as e:
                                   print("WOE")
                                   print(e)
                                   self.is_fit=False
                               except KeyError as ke:
                                   print("WOE")
                                   print(ke)
                                   self.is_fit=False
    
                   else:
                        self.is_fit=False
                        
                else:
                   self.is_fit=False
                   print("This is not an allowed encoder in this package yet.")
                   
                # return Encoder_flag
            except:
                print('an error occurs while fitting proposed encoders, please ensure the fitted data are compatible with the available encoders in this package')
                self.is_fit=False
                # return Encoder_flag
        
     # Transform all Encoder       
     def transform(self,keep_original=False):
         '''
         This function transfomrs the data by performing the recomended encoding approach
         
         Parameters
         ----------
         keep_original : bool Default=False
             This variable decides if the orignial column should be kept or deleted. The original column is deleted by default. 

         Returns
         -------
         df:    DataFrame
             It returns the updated and encoded dataframe

         '''
         # loop through fitted encoders
         if(self.is_fit):
             try:
                 # apply all transformation and append to the original data frame
                 for t,y in self.fitted_encoder.items():
                     
                     
                     if(keep_original):
                         if isinstance(y, sklearn.preprocessing.OneHotEncoder):
                             tmp_df=np.transpose(y.transform(self.df[[t]]).toarray())
                             counter=0
                             
                             for col in tmp_df:
                                 self.df[t+"_encoded"+str(counter)]=col
                                 counter=counter+1
                                 
                         else:    
                             self.df[t+"_encoded"] = y.transform(self.df[[t]])
                             
                             
                     else:
                         if isinstance(y, sklearn.preprocessing.OneHotEncoder):
                             tmp_df=np.transpose(y.transform(self.df[[t]]).toarray())
                             counter=0
                             
                             for col in tmp_df:
                                 self.df[t+"_encoded"+str(counter)]=col
                                 counter=counter+1
                                 
                             self.df = self.df.drop(t,axis=1)
                                 
                         else:    
                             self.df[t] = y.transform(self.df[[t]])
              
                 return self.df
             except:
                 print('an error occurs while performing encoding, please refit the data and apply encoders again')
         else:
             print("You need to fit the encoder first")