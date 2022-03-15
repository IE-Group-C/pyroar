# -*- coding: utf-8 -*-

from manipulator import Manipulator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler



class Scaler(Manipulator):
     def __init__(self,df,metadataO): 
       #print('Scaler')
       metadata=metadataO.copy()
       metadata = super().metadata_verifier(df,metadata)
       tmp_metadata=self.remove_no_actions_per_type(metadata,'scaling')
       super().__init__(df,metadata)
       keys = metadata.keys()
       # initialization of class variables
       self.scalers_dict = {x:tmp_metadata[x]['scaling'] for x in keys}
       self.df=df
       self.fitted_scaler={}
       self.is_fit = False

       
     def fit(self):
        '''
         This function fits the data for each scaler type of the allowed scalers in this package. 
         The allowed scalers are ['RobustScaler','StandardScaler','MaxAbsoluteScaler','MinMaxScaler']

         Manages
         -------
         is_fit : bool
            This flag confirms if all the fits were completed successfuly or there was an issue. 

        '''               
        try:
            self.is_fit=True
            # check data type
            if super().check_type(self.df[self.scalers_dict.keys()],'Numerical')==True:
                
                avaliable_transformations=['RobustScaler','StandardScaler','MaxAbsoluteScaler','MinMaxScaler']
                # check transformation availability
                if super().check_transformation(avaliable_transformations,self.scalers_dict.values(),'scaling')==True:
                    # check columns availability
                   if  super().check_columns(self.df.columns,self.scalers_dict.keys()) ==True:
                       
                       for x, y in self.scalers_dict.items():
                           
                           # check scaler type and fit
                           if y == 'RobustScaler':
                               scaler = RobustScaler()
                               self.fitted_scaler[x]= scaler.fit(self.df[[x]])
                               
                           elif y == 'MaxAbsoluteScaler':
                               scaler = MaxAbsScaler()
                               self.fitted_scaler[x]= scaler.fit(self.df[[x]])
                               
                           elif y == 'MinMaxScaler':
                               scaler = MinMaxScaler()
                               self.fitted_scaler[x]= scaler.fit(self.df[[x]])
                               
                           else:
                               scaler = StandardScaler()
                               self.fitted_scaler[x]= scaler.fit(self.df[[x]])
                               
                   else:
                       self.is_fit=False 
            
                else:
                    self.is_fit=False  
                    print("This is not one of the available scaling options")
                    
            else:
                self.is_fit=False
                print("Not all the data you are encoding are numerical")
                
            # return scaler_flag       
        except:
            print('an error occurs while fitting proposed scalers, please ensure the fitted data are compatible with the available scalers in this package')
            self.is_fit=False
            # return scaler_flag
        
        
        
           
     def transform(self,keep_original=False):
         '''
         This function transfomrs the data by performing the recomended scaling approach
         
         Parameters
         ----------
         keep_original : bool Default=False
             This variable decides if the orignial column should be kept or deleted. The original column is deleted by default. 

         Returns
         -------
         df:    DataFrame
             It returns the updated and scaled dataframe

         '''
         if(self.is_fit):
             try:
                 
                 if(keep_original):
                     for t,y in self.fitted_scaler.items():
                         self.df[t+"_scaled"] = y.transform(self.df[[t]])
                 else:
                     for t,y in self.fitted_scaler.items():
                         self.df[t] = y.transform(self.df[[t]])
                         
                 return self.df
             except:
                 print('an error occurs while performing scaling, please refit the data and apply scalers again')
         else:
            print("You need to fit the scaler first")
       