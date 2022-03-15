# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:22:41 2022

@author: jawad
"""

import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
import re
from datetime import datetime as dt
import datetime
import time #needed for unix timestamp conversion 
import copy



class Manipulator:
       
   def __init__(self,df,metadata):
       #print('General')
       # initialization of class variables
       self.df=df 
       self.metadata=metadata
       self.allowed_scaling = ['RobustScaler','StandardScaler','MaxAbsoluteScaler','MinMaxScaler','None']
       self.allowed_encoding = ['OHE','FREQ','WOE']
       self.allowed_imputation = ['ConstantImputer','MostFrequentImputer','MeanImputer','MedianImputer','None']
       self.allowed_transformations = ['reciprocal','log X','squared','None']
       self.parameters_list = ['scaling', 'encoding','impute','bin','transformation','None']
           
           
   #class methods:
   def metadata_verifier(self,df,metadata):   
       '''
       This function verifies the format of the metadata. It fills the missing information in the metadata with Nones

       Parameters
       ----------
       df : DataFrame
           The dataframe that the metadata is about.
       metadata : dict
           The metadata that is used for fit and transform functions.

       Returns
       -------
       dict
           returns the fixed metadata.

       '''
       self.metadata_temp = {}
       self.columns_list = df.columns
       self.parameters_list = ['scaling', 'encoding','impute','bin','transformation']
       self.param_val = ''
       self.current_col = {}
       
       for col in self.columns_list: 
           self.current_col = {}
           for param in self.parameters_list:
               self.param_key = param
               try:
                   self.param_val = metadata[col][param]
               except:
                   self.param_val = 'None'
   
               self.current_col[self.param_key] = self.param_val
               
           self.metadata_temp[col] = self.current_col
           
       return self.metadata_temp
    
   
   def get_recommendad_feature_engineering(self,metadata=None):
       '''
       This function is intended to print the metadata in dataframe format to make it easier for user to see the feature engineering that will 
       be done to the data.
       
       Parameters
       ----------
       metadata : dict, optional
           DESCRIPTION. This is the metadata that has the recommended feature engineering changes. The default is None which will revert ot the initialized one.

       Returns
       -------
       df: metadata in DataFrame format

       '''
       try:
           if(metadata == None):
               df = pd.DataFrame(self.metadata).transpose()
    
           else:
               df = pd.DataFrame(metadata).transpose()
           return df
       except:
           print("The metadata is either not avialble or not in the right format or not initalized.")





   def verify_modification(self,transfomration_type,transformation):
       '''
       This function checks if the proposed transformation type is allowed in this package. It checks again the lest of prepared encodings, 
       imputations, scalings, and transfomrations. 

       Parameters
       ----------
       transfomration_type : str 
           This is the type of transformation requested e.g (encoding, scaling, imputation, or transfomration).
           here are the allowed types: ['scaling', 'encoding','impute','bin','transformation']
           
       transformation : str
           This is the specific transformation requested. Here are the allowed trnasformation per type
           Scaling: ['RobustScaler','StandardScaler','MaxAbsoluteScaler','MinMaxScaler']
           Encoding: ['OHE','FREQ','WOE']
           Imputation: ['ConstantImputer','MostFrequentImputer','MeanImputer','MedianImputer']
           Transformation: ['reciprocal','log X','squared']
           
           for imputers you can pass the constant after '_'. example: ConstantImputer_100

       Returns
       -------
       bool
           returns bool indicating if transformation is allowed in this package. 

       '''
       
       try:
            if(transfomration_type.lower().strip() == 'scaling'):
                if(transformation.strip() in self.allowed_scaling):
                    return True
                else:
                    return False
            elif(transfomration_type.lower().strip() == 'encoding'):
                if(transformation.strip() in self.allowed_encoding):
                    return True
                else:
                    return False
            elif(transfomration_type.lower().strip() == 'impute'):
                if(transformation.strip().split('_')[0] in self.allowed_imputation):
                    return True
                else:
                    return False
            elif(transfomration_type.lower().strip() == 'transformation'):
                if(transformation.strip() in self.allowed_transformations):
                    return True
                else:
                    return False
       except:
            print("Your metadata changes should follow the metadata format specified in the package")
            
            
            
            
   def modify_recommended_feature_engineering(self,modified_metadata,metadata=None):
       '''
       This function is used to allow the user to modify the recommended feature engineerings that were given by the scan funciton. 
       Using this function the user can pass a dictionary that will modify only the parts specified in the modified dictionary and use the modified types if allowed 
       in the following fit and transform calls. 
       
       This funciton will not overwrite the metadata but return an updated metadata that needs to be assigned to the object.metadata

       Parameters
       ----------
       modified_metadata : dict
           this is the dictionary containing the metadata that the user would like to change in the metadata.
           
           The expected format is as follows:
               {ColumnName:{TransformationType:Tranformation}}
        e.g.
               {'Gender':{'scaling':'RobustScaler'}} 
       metadata: dict Optional
           This is the metadata that would be changed. The deafult is to modify the metadata stored already in the object. 


       Returns
       -------
       final_metadata : dict
           The function reuturns the update metadata and the user can choose to store it or not.

       '''
       try:
           if(metadata == None):
               final_metadata = copy.deepcopy(self.metadata)
           else:
               final_metadata = metadata
       except:
           print("Your metadata changes should follow the metadata format specified in the package")
           
       try:
             for k in modified_metadata.keys():
                 requested_modifications = dict(modified_metadata[k])
                 for k2 in requested_modifications.keys():
                     if(self.verify_modification(k2,requested_modifications[k2])):
                         final_metadata[k][k2] = requested_modifications[k2].strip()
                     else:
                         print(requested_modifications[k2] + ' is not an allowed '+ k2 + " type. This change is skipped.")
             return final_metadata
       except:
             print("Your modified metadata doesn't match the expected format.")
    




    
   def remove_no_actions(self,metadata):
       '''
       This function removes from the metadata all the types of transformation that were assigned 'none'
       it returns a new metadata that needs to be assigned if needed

       Parameters
       ----------
       metadata : dict
           The metadata that needs to be modified. 

       Returns
       -------
       metadata : dict
           The updated metadata.

       '''
       try:
            for k in metadata.keys():
                transfomrations = dict(metadata[k])
                for k2 in list(transfomrations.keys()):
                    if(type(transfomrations[k2]) == str):
                        if(transfomrations[k2].lower().strip() == 'none'):
                            del transfomrations[k2]
                metadata[k] = transfomrations
            return metadata
       except:
            print("Your metadata doesn't match the expected format")
    

    


   def remove_no_actions_per_type(self,metadata,transformation_type):
       '''
       This function removes the entier key (equivelatn to column in this context) that the recommended transformation_type for it is none from the dataframe.

       Parameters
       ----------
       metadata : dict
           Metadata that needs to be checked.
       transformation_type : str
           the transformation type that we would like to check. example 'scaling'.

       Returns
       -------
       metadata : dict
           The updated metadata after removing the keys that have none for this transformation type.

       '''
       try:
            for k in list(metadata.keys()):
                transfomrations = dict(metadata[k])
                if(type(transfomrations[transformation_type]) == str):
                    if(transfomrations[transformation_type].lower().strip() == 'none'):
                        del metadata[k]
            return metadata
       except:
            print("Your metadata doesn't match the expected format")



    
   
   def is_float(self,val):
        '''
           this function compares if val array is numerical or categorical, it will return true or false per item
    
           Parameters
           ----------
           val : numerical
               value to check.
    
           Returns
           -------
           bool
               val is numerical or not.

        '''
       # this function compares if val array is numerical or categorical, it will return true or false per item
        try:
            float(val)
        except ValueError:
            # print("The value expected is float")
            return False
        else:
            return True
        
        
   
   def check_columns(self,df_columns,proposed_columns):
       '''
       This function wether two list contain similar items for columns amd transformations

       Parameters
       ----------
       df_columns : list
           list of columns in the dataframe.
       proposed_columns : str
           the column that need to be checked.

       Returns
       -------
       bool
           boolean value indicating if the column exists or not.

       '''
       try:
           if len(proposed_columns)==0  :
               print('No column was proposed')
               return False
           else: 
               for i in proposed_columns:
                   
                   if i not in df_columns:
                       print(i,' column is not exist in the current dataframe ')
                       return False
               return True
       except:
            print("The proposed columns or df_columns are not formatted as expected")
       
        
 
    
   def check_transformation(self,current_transformation,proposed_transformation,operation):
       '''
       This function checks if the prposed transformation type in the metadata is a valid transformation in the package

       Parameters
       ----------
       current_transformation : list
           The current transfomration that is in the dataframe.
       proposed_transformation : list
           The newly proposed transformation.
       operation : str
           The type of operation e.g (encoding, scaling, imputation, or transfomration).

       Returns
       -------
       bool
           bool value if the transfomration is allowed or not.

       '''
       try:
           if len(proposed_transformation)==0  : 
               print('no',operation, 'was proposed')
               return False
           else: 
               
               for i in proposed_transformation:
                   t=False
                   for o in current_transformation:
                       ab= '^'+o
                       regexp = re.compile(ab)
                       if regexp.search(i):
                           t=True
                   if t == False:
                       print(i, operation ,'is not exist in the current package ')
                       return False
               return True
       except:
           print("Check Transformation ran into an error while checking the proposed transfomration")
       
        

    
   # check data type if metadata is modified by the user 
   def check_type(self,df,data_type):
       '''
       This function checks data type if metadata is modified by the user 

       Parameters
       ----------
       df : DataFrame
           The dataframe to use.
       data_type : str
           The data type to check for.

       Returns
       -------
       bool
           The result indicating if the type is as expected.

       '''
       try:
           a= np.vectorize(self.is_float)(df)
           if data_type=='Numerical':
              # use the product of numpy array , if equal 1 means all numerical
              if np.prod(a) != 1:
                  df_t = pd.DataFrame(np.prod(a,axis=0).reshape(1,len(df.columns)), columns = df.columns)
                  print('Ensure all columns are Numerical , please refer to the below dataframe for more info')
                  print(df_t)
                  return False
              else:
                  return True
           # string operations can be apply later 
           elif data_type=='Categorical':
              # use the sum of numpy array , if less than 1 means all numerical 
              if np.sum(a) >= 1 :
                  df_t = pd.DataFrame(np.prod(a,axis=0).reshape(1,len(df.columns)), columns = df.columns)
                  print('Ensure all columns are Categorical, please refer to the below dataframe for more info')
                  print(df_t)
                  return False
              else:
                  return True
           else: 
               print(data_type,'is unknown please select Numerical or Categorical')
       except:
           print("We ran into an issue checking the type in function check_type")




#Start of Date function:--------------------------------------------------------------------------------------
   def date_formatting(self,d, century = '20'):
       '''
           This function checks and correct the date format to be appicable for date operations
               The function accepts the follwing formats
               yyyy/mm/dd
               yyyymmdd
               yymmdd
               yyyy/m/d
               31 Jan 2022
               Note: the simbol "/" could be any kind of character separator. 
           
           Parameters
           ----------
            d : str
                string variable indicating the date 
                
            century : str Default:'20'
                the cnetury we are using
           
           Returns
           -------
           bool
               boolean value indicating if the column exists or not.
       '''
   
       def date_alpha_cleaning(self,d):
           month_text = ''
           month_num = 0
           months_l = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                         'november', 'december']
           months_s = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
           result_date = ''
   
           if len(d) == 9:
               month_text = d[2:5]
           else:
               month_text = d[2:len(d)-4]
   
           if month_text in months_l:
               month_num = int(months_l.index(month_text) + 1)
           elif month_text in months_s:
               month_num = int(months_s.index(month_text) + 1)
           else:
               return -1
   
           if month_num < 10:
               month_num = str('0' + str(month_num))
           else:
               month_num = str(month_num)
   
           result_date = d[-4:] + month_num + d[:2]
           return int(result_date)
   
    
   
    
   
    
   
       def remove_date_separators(self,d):
           #Check the position of the separator:
           date_structure = []
           date = ''
           sp = 0
           fount_first_sp = False
           fount_second_sp = False
           month_deg = 0
           day_deg = 0
   
           try:
               for i in range(len(d)):
                   try:
                       int(d[i])
                       date_structure.append('int')
                       date = date + str(d[i])
                       if fount_first_sp and not fount_second_sp:
                           month_deg += 1
                       elif fount_first_sp and fount_second_sp:
                           day_deg += 1
                   except:
                       try:
                           str(d[i])
                           date_structure.append('char')
                           if not fount_first_sp:
                               fount_first_sp = True
                           elif fount_first_sp:
                               fount_second_sp = True
                       except:
                           return -1
   
               #Number separators:
               sp = sum(map(lambda x: x == 'char', date_structure))
               if sp != 0 and sp != 2:
                   return -1
               else:
                   return str(date), month_deg, day_deg
   
           except:
               return -1
       
       # Start of the main function: -----------------------------------------------------------------
       #d_temp_int = 0
       d_temp_str = ''
       d_result = dt.date
       #current_year = 0
       error = False
       d_alpha_cleaned = 0
       d_without_sp = 0
       d_fixed = ''
       month_deg = 0
       day_deg = 0
       
       if len(century) > 2:
           error = True
       else:
           
           #Remove separators and perform format cleaning:
           d = d.replace(" ", "").lower()
           
           #alphabitical month cleaning:
           d_alpha_cleaned = date_alpha_cleaning(d)
           
           if d_alpha_cleaned == -1:
               d_without_sp, month_deg, day_deg = remove_date_separators(d)
               
               if d_without_sp == -1:
                   error = True
               else:
                   d_fixed = str(d_without_sp)
           else:
               d_fixed = str(d_alpha_cleaned)
           
           #Check if error reported from format cleaning function:
           if error != True or len(d_fixed) <= 10:
               #Check if the string can be converted into a number
               try:
                   d_temp_str = str(d_fixed)
   
                   if len(d_fixed) == 4:                    
                       d_result = datetime.date(int(century + d_temp_str[:2]), int(d_temp_str[2]), int(d_temp_str[3]))
                   elif len(d_fixed) == 5:
                       if day_deg == 2:
                           d_result = datetime.date(int(century + d_temp_str[:2]), int(d_temp_str[3]), int(d_temp_str[-2:]))
                       else:
                           d_result = datetime.date(int(century + d_temp_str[:2]), int(d_temp_str[2:4]), int(d_temp_str[-1:]))
                   elif len(d_fixed) == 6: 
                       d_result = datetime.date(int(century + d_temp_str[:2]), int(d_temp_str[2:4]), int(d_temp_str[4:]))
                   elif len(d_fixed) == 7:
                       d_result = datetime.date(int(d_temp_str[:4]), int(d_temp_str[4]), int(d_temp_str[-2:]))
                   elif len(d_fixed) == 8:
                       d_result = datetime.date(int(d_temp_str[:4]), int(d_temp_str[4:6]), int(d_temp_str[6:]))
                   else:
                       error = True
   
                   if not error:  
                       #print(d_result)
                       return d_result
               except:
                   error = True
   
       if error:
           return -1 #Error
       
        
       
   def date_diff(self,d1,d2,unit = 'days'):
       '''
        This function calculates the difference between two dates, it cleans it up first using a custom fucntion 
        "date_formattting" and then returns the difference based on user's request in days, months or years
        
        Parameters
        ----------
         d1 : date
             the 'from' date
         d2 : date
             tthe 'to' date 
        
        unit: str   Default: days
            the unit to use 'days' or 'months' or 'years'
        
        Returns
        -------
        duration
            the difference based on user's request in days, months or years
       '''
       try:
           #Variables:
           duration = 0
           
           #Perform Date validation and cleaning:
           d1 = self.date_formatting(d1)
           d2 = self.date_formatting(d2)
           
           if d1 == -1 or d2 == -1:
               raise -1 #Error
           
           #Perform Duration Calculation:
           if unit == 'days': 
               duration = (d2-d1).days
           elif unit == 'months': 
               duration = (d2-d1).days/30
           elif unit == 'years': 
               duration = (d2-d1).days/365.25
           else:
               return -1 #Invalid duration unit
           
           return duration
       except:
           return -1
       
        
       
       
   def unix_time(self,d1): 
       '''
       This function to clean and convert datetime to a unix timestamp
       This function is needed to be called from date_transform
       '''
       return time.mktime(self.date_formatting(d1).timetuple()) #as unix timesstamp
   
   
   
   def time_diff_unix(self,d1,d2): 
       '''
       compares the difference between two dates in unix timestamp
       example use of function: 
       time_diff_unix('30 nov 2022',1735506000)
       '''
       return (self.unix_time(d1)-d2)
   
    
   
    
   def date_transform(self,d1,d2=None,unit = 'days',time_zone = "America/Los_Angeles"):
       '''
       

       Parameters
       ----------
       d1 : TYPE
           DESCRIPTION.
       d2 : TYPE, optional
           DESCRIPTION. The default is None.
       unit : TYPE, optional
           DESCRIPTION. The default is 'days'.
       time_zone : TYPE, optional
           DESCRIPTION. The default is "America/Los_Angeles".

       Returns
       -------
       TYPE
           DESCRIPTION.

       '''
       
       #checking what version of the polymorphic function to use 
       if d2 is  None : #one column provided
           return np.vectorize(self.unix_time)(d1) #return unittime 
       
       elif isinstance(d2,pd.Series) : #two columns version
           return np.vectorize(self.date_diff)(d1.astype(str),d2.astype(str),unit) #change this 
       
       elif isinstance(d2, int) : #compared to a static number (unix timestamp)
           return np.vectorize(self.time_diff_unix)(d1.astype(str),d2) #returns difference in unixtimestamp   
       
       elif isinstance(d2,str)  : #compare to a static date inserted as string
           return np.vectorize(self.date_diff)(d1.astype(str),d2,unit)
       else :
           return 'idk'
   
   def validate_saudi_national_id(self,snid):
       '''
       This function takes a national id and verifies if it is a valid number for Saudis

       Parameters
       ----------
       snid : str
           national id.

       Returns
       -------
       message
           returns a message that either the national id is valid or not.

       '''
       
       temp = ''
       total = 0
       try:
           snid = str(snid)
   
           if snid[0] != '1' and snid[0] != '2':
               return 'Error'
   
           if len(snid) != 10:
               return 'Error'
   
           for i in range(1,len(snid)):
               if i % 2 != 0:
                   temp = str(int(snid[i-1]) * 2)
                   if len(str(temp)) == 1:
                       temp = str(str('0') + str(int(int(snid[i-1]) * 2)))
   
                   total = total + int(temp[0]) + int(temp[1])
               else:
                   total = total + int(snid[i-1])
                   
           if str(total)[1] == snid[-1:] or int(snid[-1:]) == ( 10 - int(str(total)[1])):
               return 'Valid ID'
           else:
               return 'Not Valid ID'
           
       except:
           return 'Error'
       

#End of Date function:--------------------------------------------------------------------------------------

   def categorical_regrouping(self, df, thresh=0.03, name= 'Other'):
       '''
       This function takes all object type columns and checks for values that have occurance less than the specified threshold and group these values into one value
       The function is helpful for ML preparation to avoid having rare values appearing only on training after the train-test split
       It also helps reducing the number of columns in case of OHE for a categorical variable
       It takes 3 parameters:
           1- df = dataframe name
           2- thresh = the percentage of how frequent the values occurance that should be regrouped (optional) with default value as 0.03
           3- name = the name where the values under the threshold will be changed to (optional) with default value as 'Other'
       '''
       regrouped = []
        
       cat_features = df.select_dtypes(include=np.object).columns.tolist()
       for col in cat_features:
            check_values= df[col].value_counts(normalize=True).reset_index()
            check_values.columns=[col, 'count percentage']
            
            for i in range(len(check_values)):
                if(check_values['count percentage'][i] < thresh):
                    regrouped.append(check_values[col][i])
            df[col] = np.where(df[col].isin(regrouped),name,df[col])
       return df
            