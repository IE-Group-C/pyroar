
from pyroar import Pyroar
import pandas as pd
from sklearn.model_selection import train_test_split
##############  MAIN #############
#from sklearn import preprocessing

loan_df = pd.read_csv('AUTO_LOANS_DATA.csv', sep = ';')


# initialize list of lists
data = [['M', 10,1], ['F', 15,1], ['M', 14,0], ['F',22,1], ['M',30,1], ['F',35,0], ['F',40,0]]
 
# Create the pandas DataFrame
my_df = pd.DataFrame(data, columns = ['Gender', 'Age', 'Result'])
 
# print dataframe.
my_df

#using loan instead: 
my_df = loan_df

X = pd.DataFrame(my_df.drop(['SEX'],axis=1))
y = pd.DataFrame(my_df['SEX'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)


    
ct = Pyroar()
ct.printDf()

#meta = ct.scan(my_df, 'Result')  #test dataframe 
meta = ct.scan(loan_df, 'SEX', dropThreshold=0.9) #loan dataframe

ct.printDf()

print("strat: prep ")
ct.prep()
ct.printDf()

print("start: fit")
ct.fit(X_train, y_train)
ct.printDf()

print("start: transform")
ct.transform(X)
ct.printDf()

print("End ----")













