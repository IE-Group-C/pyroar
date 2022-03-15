# -*- coding: utf-8 -*-
from manipulator import Manipulator
import numpy as np


class Transformer(Manipulator):
     def __init__(self,df,metadataO):
       metadata=metadataO.copy()
       metadata = super().metadata_verifier(df,metadata)
       tmp_metadata=self.remove_no_actions_per_type(metadata,'transformation')
       super().__init__(df,metadata)
       keys = metadata.keys()
       # initialization class variables
       
       self.df=df
       self.transformation_dict = {x:tmp_metadata[x]['transformation'] for x in keys}
       
       
               
      
     def transform(self,keep_original=False):
            '''
             This function transfomrs the data by performing the recomended transformation approach

             The allowed transformers are ['reciprocal','log X','squared']
             
             Parameters
             ----------
             keep_original : bool Default=False
                 This variable decides if the orignial column should be kept or deleted. The original column is deleted by default.  
    
             Returns
             -------
             transform_flag : bool
                 This flag confirms if all the transformation were completed successfuly or there was an issue.
                 
             df:    DataFrame
                 It returns the updated and transformed dataframe
            '''    
        # fit all numerical transformations
            try:
                transform_flag=True # to indicate if transform is completed successfully  
                
                avaliable_transformations=['reciprocal','log X','squared']
                # check transformation availability
                if super().check_transformation(avaliable_transformations,self.transformation_dict.values(),'transformation')==True:
                # check columns availability
                   if  super().check_columns(self.df.columns,self.transformation_dict.keys()) ==True:
                       
                       for x, y in self.transformation_dict.items():
                           if super().check_type(self.df[[x]],'Numerical')==True:
                               
                               if(keep_original):
                                   # check numerical transformations and fit
                                   if y == 'reciprocal':
                                      self.df[x+"_reciprocal"] = np.reciprocal((self.df[x].replace(0,1)))
        
                                   elif y == 'log X':
                                       self.df[x+"_logX"] = np.log(self.df[x].replace(0,1))
                                       
                                   elif y == 'squared':
                                      self.df[x+"_squared"] = np.power((self.df[x]),2)                              
                                          
                                   else:
                                       transform_flag=False
                                       print("the tnrasformation is not of an allowed type")
                               #drop the original column
                               else:
                                    # check numerical transformations and fit
                                    if y == 'reciprocal':
                                       self.df[x] = np.reciprocal((self.df[x].replace(0,1)))
         
                                    elif y == 'log X':
                                        self.df[x] = np.log(self.df[x].replace(0,1))
                                        
                                    elif y == 'squared':
                                       self.df[x] = np.power((self.df[x]),2)                              
                                           
                                    else:
                                        transform_flag=False
                                        print("the tnrasformation is not of an allowed type")
                                      
                           else:
                               transform_flag=False
                               print("The column is not of numerical type")
                          
                   else:
                        transform_flag=False
                else:
                   transform_flag=False
                   print("This trasformation type is not allowed")
                return transform_flag, self.df 
            except:
                print('an error occurs while performing numerical transformation, please ensure the fitted data are compatible with the available transformation in this package')
                transform_flag=False
                return transform_flag, self.df          