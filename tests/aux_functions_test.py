# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:20:23 2022

@author: fagfa
"""

from datetime import datetime as dt
import datetime
import time #needed for unix timestamp conversion 
import numpy as np
import pandas as pd

#from src.pyroar.aux_functions import aux_functions 

#import src.pyroar.aux_functions.aux_functions 

from aux_functions import date_formatting, date_transform
#---------------------------- Test Functions ----------------------------------

#Start the Unit Functional Test:-----------------------------------------------
print('************** Test **************')

# 1) Test Date formatting function:
print('[1] Test date formatting function:--------------------------------------')

print('Case[1]: convert [2022929] to [2022-09-29]')
test = date_formatting('2022929')
if test == datetime.date(2022, 9, 29):
   print('Passed successfully.')
else:
   print('Failed')

print('-------------')
print('Case[2]: convert [30 nov 2022] to [2022-11-30]')
test = date_formatting('30 nov 2022')
if test == datetime.date(2022, 11, 30):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[2]: convert [20221029] to [2022-10-29]')
test = date_formatting('20221029')
if test == datetime.date(2022, 10, 29):
   print('Passed successfully.')
else:
   print('Failed')
   
   
print('-------------')
print('Case[3]: convert [2022.10.29] to [2022-10-29]')
test = date_formatting('2022.10.29')
if test == datetime.date(2022, 10, 29):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[4]: convert [2022/10/29] to [2022-10-29]')
test = date_formatting('2022/10/29')
if test == datetime.date(2022, 10, 29):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[5]: convert [22/10/20] to [2022-10-20]')
test = date_formatting('22/10/20')
if test == datetime.date(2022, 10, 20):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[6]: convert [2022-10-20] to [2022-10-20]')
test = date_formatting('2022-10-20')
if test == datetime.date(2022, 10, 20):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[7]: convert [22.10/2] to [2022-10-02]')
test = date_formatting('22.10/2')
if test == datetime.date(2022, 10, 2):
   print('Passed successfully.')
else:
   print('Failed')


print('-------------')
print('Case[7]: convert [22.1/2] to [2022-01-02]')
test = date_formatting('22.1/2')
if test == datetime.date(2022, 1, 2):
   print('Passed successfully.')
else:
   print('Failed')

print('Test date formatting function completed successfully--------------------\n\n')


# 2) Test polymorphismness of date_transform function:
print('[2] Test polymorphismness of date_transform function :------------------')

dates_data = [['2016-02-08', '2016-03-01'], ['2016-04-01', '2016-05-01'],
              ['2016-06-01', '2016-09-01'],
              ['2016-11-01', '2017-01-01'],
              ['2017-02-01', '2017-03-01'], ['2017-04-01', '2017-05-0']]

df_dates = pd.DataFrame(dates_data, columns = ['start_date', 'end_date'])


#case 1: only one column provided 
answer = date_transform(df_dates['start_date'])#converts the column into unix timestamp
print('case1',answer)

#case 2: two columns provided
# this finds the difference between the two columns in the unit provided "years" in this case
answer = date_transform(df_dates['start_date'],df_dates['end_date'],'years') 
print('case2',answer)

#case 3: a column and a unix timestamp provided 
# returns the difference between the column and the other 
answer = date_transform(df_dates['start_date'],1735506000)
print('case3',answer)

#case 4: a column and a constant date 
#return the difference between each element in the column and a given constant date in the unit requested
answer = date_transform(df_dates['start_date'],'2020-11-29','months') 
print('case4',answer)



















