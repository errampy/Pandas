import pandas as pd

def int_data():
    a=[1,2,3]
    s=pd.Series(a)
    # print('s\n\n',s)
    print('type of s ',type(s)) # <class 'pandas.core.series.Series'>
    print(s)
    # here dtype is int64 

# int_data()

def str_data():
    a=['python','java','pandas']
    s=pd.Series(a)
    print(s)
    # here dtype is object becase for string value dtype consider as a objer

# str_data()
    
def float_data():
    salaries_list = [1200.3,123030.7,12388.20]
    s=pd.Series(salaries_list)
    print(s)
    # here dtype is float64 of value is float

# float_data()
    
def dict_data():
    a={0:'Python',1:'Java',2:'Pandas'}
    s=pd.Series(a)
    print(s)
    # here dtype is object

# dict_data()
import numpy as np
def various_data():
    s1=pd.Series(data=10)
    s2=pd.Series(data=[102,39,49,38])
    s3=pd.Series(data={1:1,2:2,3:3,4:'Four'})
    s4=pd.Series(data=np.array([10,20,30,40]))
    s5=pd.Series(data='ram')
    print(s1)
    print(s2)
    print(s3)
    print(s4)
    print(s5)

# various_data()
    
# Series argument
    
def series_parameter():
    a=[10,20,30,40]
    custome_index=a
    s=pd.Series(data=a,dtype='float',index=custome_index,name='Custome',copy=True)
    s.iloc[0]=50
    print(s)
    print(a)
# series_parameter()

def get_default_range_index():
    '''
    If we are not providing index parameter , thenpandas will consider default index values from RangeIndex(0,1,2...,n) boject internally.
    '''
    a=[10,20,30]
    s=pd.Series(data=a)
    print('index ',s.index)

# get_default_range_index()
    
def change_range_index():
    name_list = ['ram','mohan','sohan','sumit']
    s = pd.Series(data=name_list)
    s.index=pd.RangeIndex(start=10,stop=14,step=1)
    print(s)
# change_range_index()
    

# =*=*=*=*=*=*=* Excercise =*=*=*=*=*=*=*
'''
Question
1. Create a python list named with students_list with 5 studet names?
2. create another python list named with marks_list with corresponding student marks?
3. create a Series object that stores studens marks as values and student names as index labels.
   assign name 'studens' for this series
4. Create a python dict with students_list and marks_list.create a series object with that dictionary?
'''    

def convert_list_dict(students_list,marks_list):
    '''
    convert two list into dict obj
    '''    
    my_dict={}
    for i in range(len(students_list)):
        my_dict[students_list[i]]=marks_list[i]
    return my_dict
def for_python_dict(students_list,marks_list,series_name):
    '''
    4. Create a python dict with students_list and marks_list.create a series object with that dictionary?
    '''
    converted_dict=convert_list_dict(students_list,marks_list)
    for_dict_ser_obj =pd.Series(data=converted_dict,name=series_name)
    return for_dict_ser_obj
def excercise(students_list,marks_list,series_name):
    '''
    1. Create a python list named with students_list with 5 studet names?
    2. create another python list named with marks_list with corresponding student marks?
    3. create a Series object that stores studens marks as values and student names as index labels.
    assign name 'studens' for this series
    '''
    s=pd.Series(data=students_list,index=marks_list,name=series_name)
    return s

# print('ENTER THE VALUE')
# students_list=eval(input('enter name in list format like ["Ram","mohan"]'))
# marks_list=eval(input('Enter mark in list format like [60,50]'))
# series_name=input('series name like RamSeries ')

# print('LIST ANSWER')
# print(excercise(students_list,marks_list,series_name))
# print('DICT ANSWER')
# print(for_python_dict(students_list,marks_list,series_name))
# print('*'*20+'Thank You!'+'*'*20)


# ======================== head and tail ============================

def head_in_series():
    s = pd.Series(data=[i for i in range(30)],name='head_in_series')
    print('s ',s)
    # user head()
    print(s.head())# by defauld head will return first 5 rows
    print(s.head(3))# here head will return first 3 rows , that meas we can customize it
    # nevative value
    print(s.head(-20))# it will return all except last 20 rows

# head_in_series()

def tail_in_series():
    s = pd.Series(data=[i for i in range(30)],name='tail in series')
    print('s ',s)
    print(s.tail())# by default tail will return last 5 rows
    print(s.tail(3)) # here tail will return last 3 rows , that means we can customized it too
    # negative value
    print(s.tail(-20))# here tail will return all the value from the series object except first 20

# tail_in_series()
    
# =*=*=*=*=*=*=* Excercise =*=*=*=*=*=*=*
'''
i have series object there contain there upto 30 so i wants data 10-15 from that only , by using head and tail
'''

def head_tail_excercise():
    s = pd.Series(data=[i for i in range(30)],name='HEAD TAIL EXCERCISE')
    # get first 15
    first_15 = s.head(15)
    print('first 15 ',first_15)
    # lets get last 5
    result = first_15.tail()
    print('OUR RESULT IS ')
    print(result)

# head_tail_excercise()
    
# =============== Extraction Start=======================
# Extract values from Series by using index position:
"""
Syntax :
     s[x]

     x can be index value
     x can be list of indices 
     x can be slice also

Example : 
    s[5] ==> Returns value present at index 5
    s[[1,3,5]] ==> Returns values present at indices 1,3 and 5.
    s[2:5] ==> Return Series of values from 2nd index 4th index.
"""
# 

def alphate_by_user():
    alphabets=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print(alphabets)
    return alphabets

# alphate_by_user()
from string import ascii_uppercase
def alphabets_from_strings():
    alphabets = list(ascii_uppercase)
    # print(alphabets)
    return alphabets
# alphabets_from_strings()

def our_series():
    s = pd.Series(data=alphabets_from_strings(),name='ALPHABETS')
    # print(s)
    return s

# our_series()
    
def get_first_character():
    s = our_series()
    print(s[0])

# get_first_character()
    
def get_last_character():
    s = our_series()
    print(s[25])

# get_last_character()
    
def get_last_character_dynamically():
    s = our_series()
    print(s[s.size-1])

# get_last_character_dynamically()

def get_character_at_indices():
    s = our_series()
    print(s[[4,8,25]])

# get_character_at_indices()
    
def get_character_from_and_to_indeices():
    s = our_series()
    print(s[10:26])

# get_character_from_and_to_indeices()
    
def get_character_alternative():
    s = our_series()
    print(s[::2])

# get_character_alternative()
    
def get_first_5_character():
    s = our_series()
    print(s[:5])

# get_first_5_character()
# note: -ve indexing is applicable only for  slice input
'''
s[-1]  ===> invalid
s[[-1,-2,-3]]  ==> invalid
s[-3:]  ==> valid
'''

# Note : accessing based on position is application even for custom labeled series also

# *****************************************************
# Extrace values from Series by labels
'''
Syntax:
    s['label']
    s[['label-1','label-2','label-3']]
    s[label1:label2]
'''

# =============== Example==============================

def add_label_using_map(sequence):
    seq = map(lambda x : 'Label_'+x,sequence)
    return list(seq)
# print('add_label_using_map ',add_label_using_map(our_series()))
def add_label_using_list_comprehension(sequence):
    print('Here',['Label_'+i for i in sequence])
# print(add_label_using_list_comprehension(our_series()))
    
def adding_prefix():
    s = pd.Series(data=alphabets_from_strings(),index = alphabets_from_strings(), name='ALPHABETS')
    # print(list(map(lambda x : 'Label_'+
    #                x,s)))
    s = s.add_prefix('Label_')
    return s
# print(adding_prefix())

def adding_suffix():
    s = pd.Series(data=alphabets_from_strings(),index = alphabets_from_strings(), name='ALPHABETS')
    # print(list(map(lambda x : 'Label_'+
    #                x,s)))
    s = s.add_suffix('_Label')
    return s
# print(adding_suffix())
# map concepts
'''
it always expecting some function as a argument

map(function,sequence)

'''
# =============== Extraction End=======================

# Extracting values by using loc and iloc indexers
# ------------------------------------------------
'''
loc indexer ==> For label based selection
iloc iindexer ==> For Position based selection ==> i-> integer
'''

# iloc indexer.....
def iloc_indexer():
    s = pd.Series([10,20,30,40,50])
    # print(s)

    # Example
    # Q1. To get first character?
    # s[0] or s.iloc[0]
    print(s.iloc[0])# access by single index
    # To get the values as indies 0,1
    print(s.iloc[[0,1]])# access by multiple index
    # get first 2 value 
    print(s.iloc[:2])# access by slicing

    # get last values
    # print(s[-1])# invalid
    print(s.iloc[-1]) # valid

# loc indexer
'''
For label based selection

Syntax :
--------
s.loc[label]
s.loc[label1,label2,label3]
s.loc[labelm:labeln]# here labeln is inclusive
'''
def loc_indexer():
    alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    s = pd.Series(data=alphabets,index=alphabets)
    s = s.add_prefix('Label_')
    print(s)
    # get the value associated with Label_A
    print(s.loc['Label_A'])
    # get the values associated with Label_A, Label_K, Label_Y
    print(s.loc[['Label_A','Label_K','Label_Y']])
    # To get the values from Label_H to Label_N?
    print(s.loc['Label_H':'Label_N'])
    # To get the alternative values from Label_H to Label_N 
    print(s.loc['Label_H':'Label_N':2])


# loc_indexer()
    
# Boolean Making For Condition Based Selection:
'''
Condition Based Selection
We have to provide array of boolean values and selects values from Series where True value Present
It is application for normal indexer , loc and iloc indexer also.

Syntax:
--------
s[[True,False,....]]
s.loc[[True,False,....]]
s.iloc[[True,False,....]]
s.get([True,False,....])
'''

def boolean_masking():
    s = pd.Series([10,20,30,40,50])
    print(s[[True,False,False,True,True]])
# boolean_masking()
'''
***Note : The number of boolean values passed and the number of values in Series
must be matched , otherwise we will get error.
# IndexError: Boolean index has wrong length: 6 instead of 5

But In the case of get() method we won't get any error and just we will get None
For Example:


def get_boolean_masking():
    s = pd.Series([10,20,30])
    print(s.get([True,False,True,False]))# None
get_boolean_masking()

***Note : This approach  is very helpful to get values basesd on some condition.
'''

# Q. To Select all values which are > 25
def select_value_gt_25():
    s = pd.Series([10,20,30,40,50])
    print(s.loc[s>25])
    print(s[s>25])
# select_value_gt_25()

# Q To Select the value which are divisible by 3
def select_value_div_by_3():
    s = pd.Series([i for i in range(20)])
    print(s[s%3==0])
# select_value_div_by_3()
    
# Usages of callables in selecting Elements:
#-------------------------------------------
'''
We can use Callable object like function while selectionvalues from the Series.
It should return anything , which be valid argument for indexers and methods
'''

def odd_selection(s):
    return [True if i%2==1 else False for i in range(s.size)]

s = pd.Series([i for i in range(20)])

# print(s[odd_selection])
# print(s.loc[odd_selection])
# print(s.iloc[odd_selection])
# print(s.get([odd_selection]))

'''
We can pass callable object to normal indexer,loc,iloc indexer and gor get() method also.
'''
# Q. Select all values where salary is in between 2500 to 5000

# solution
def get_salary_between_range():
    s = pd.Series(
    data=[1000,2000,3000,4000,5000,6000],
    index=['ram','mohan','sohan','raju','raji','ranjan'])
    print(s[lambda s : [True if sal >= 2500 and sal <=5000 else False for sal in s]])
# get_salary_between_range()
    
'''
Summary : How to get values from Series objects:
-----------------------------------------------
s.head(n)
s.tail(n)
s[index]
s[index1,index2,index3,...]
s[indexm:indexn]

s[label]
s[label1,label2,...]
s[labelm:labeln] # here labeln is inclusive

s.iloc[index]
s.iloc[index1,index2,...]
s.iloc[indexm:indexn] # here labeln is inclusive

s.loc[label]
s.loc[label1,label2,..]
s.loc[labelm:labeln] # here labeln is inclusive

s.get(index)
s.get([index1,index2,index3,...])

s.get(label)
s.get(['label1','label2])

# in get slicing is not supported...
Even we can provide boolean mask values and callable objects as arguments.


1 telari => 4000 ==> 1*3 = 12,000
1 kuntal ==> 6000 ==> 1*5 = 30,000

'''

# The Important attributes of Series Object :
#--------------------------------------------
'''
Attribute are nothing but properties which provides information
about the Series object.

The following are various important attributes of Series object.
1. s.values:
----
Return values present inside the Series objects
Example:
def s_values():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.values)
s_values()

Output:
['Sunny' 'Bunny' 'Chinny' 'vinny' 'Pinny' 'Kikhill']

2. s.values:
----
Return index  the Series objects

Example :

def s_index():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.index)
s_index()

Output:
Index([10, 20, 30, 40, 50, 60], dtype='int64')

3. s.dtype:
----
Return index  the Series objects

Example:

def s_dype():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.dtype)
s_dype()

Output:

object

4. s.size:
----
Return the number of elements presents in the Series objects

def s_size():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.size)
s_size()

Output:
6


5. s.shape:
----
Returns a tuple of the shape of underlying data.
in the case of Series, it is single valued tuple , which presents the number of 
elements presents in the series
(12,) ==> ID
(2,3) ==> 2D

Example:

def s_shape():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.shape)
s_shape()

Output:

(6,)

6. s.ndim:
----
Return number of dimention of the underlying data, by definiation 1

Exampmle:


def s_ndim():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.ndim)
s_ndim()

Output:
----
6

7. s.name:
----

Example :
def s_name():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.name)
s_name()

Output:
None

8. s.is_unique:
----
Returns True if value of the objects are true

Example:
def s_is_unique():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill'],
        index=[10,20,30,40,50,60]
    )
    print(s.is_unique)
s_is_unique()

Output:
True

Note : 
To get number of unique value , we can use nunique method.
By default it ignore(drops) NaN values, if we want to consider NaN values 
also then we have to user dropna=False
Example:


def s_nunique():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill','Sunny',np.NaN,],
        index=[10,20,30,40,50,60,70,80]
    )
    print(s.nunique)
s_nunique()


Output: 
def s_nunique():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill','Sunny',np.NaN,],
        index=[10,20,30,40,50,60,70,80]
    )
    print(s.nunique)
s_nunique()


Output:

<bound method IndexOpsMixin.nunique of 10      Sunny
20      Bunny
30     Chinny
40      vinny
50      Pinny
60    Kikhill
70      Sunny
80        NaN
dtype: object>

Another Example :
def s_nunique():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill','Sunny',np.NaN,],
        index=[10,20,30,40,50,60,70,80]
    )
    print(s.nunique())
s_nunique()

output: 6

Another Example:

def s_nunique():
    s = pd.Series(
        data=['Sunny','Bunny','Chinny','vinny','Pinny','Kikhill','Sunny',np.NaN,],
        index=[10,20,30,40,50,60,70,80]
    )
    print(s.nunique(dropna=False))
s_nunique()

Output:7

9. s.is _monotonic is_monotonic_increasing  is_monotonic_decreasing:
-------------------------------------------------------------------
monotoni means whether values are in some order or not like ascending order or descending order etc...

s.is_monotoni attribure Returns True if values in the object are monotonic_increasing.

Example:

def s_is_monotonic():
    s1 = pd.Series([10,20,30,40])
    s2 = pd.Series([40,30,20,10])
    s3 = pd.Series([10,20,40,30])
    print(s1.is_monotonic_increasing) # True
    print(s2.is_monotonic_decreasing) # True
    print(s3.is_monotonic_decreasing) # False
    print(s3.is_monotonic_increasing) # False
    
s_is_monotonic()




10. s.hasnans:
---------------
Returns True if Series contains NaN or None,
ie we can use this attribute to check whether some values are absent/missing or not.

Example:

def s_hasnans():
    s1 = pd.Series([10,20,30,40])
    s2 = pd.Series([40,30,20,10,pd.NA])
    print(s1.hasnans) # False
    print(s2.hasnans) # True
    
s_hasnans()

.....


'''
import numpy as np

def s_hasnans():
    s1 = pd.Series([10,20,30,40])
    s2 = pd.Series([40,30,20,10,pd.NA])
    print(s1.hasnans) # False
    print(s2.hasnans) # True
    
# s_hasnans()
    
# Passing Series object to the Python's inbult function
'''
We can pass pandas Series object to python's inbuit functions.

1. len(s)
It returns the number of elements present in the Series objects

Example:
def s_lens():
    s = pd.Series([10,20,30,40])
    print(len(s)) #4
    
s_lens()

2. type(s):

Example:

def s_types():
    s = pd.Series([10,20,30,40])
    print(type(s)) # <class 'pandas.core.series.Series'>
    
s_types()


3. dir(s):
It reutrns a list of alll members(varialble and methods) which are applicable for series object

Example
def dir_s():
    s = pd.Series([10,20,30,40])
    print(dir(s)) # <class 'pandas.core.series.Series'>
    
dir_s()

4. sorted(s):

it will sort the elements present into Series objects and returns List of those values
def sorted_s():
    s = pd.Series([10,20,30,40])
    print(sorted(s)) # <class 'pandas.core.series.Series'>
    
sorted_s()


6. list(s):
---------------
To get series object values in the form of list, it is series to list conversion.
def list_s():
    s = pd.Series([10,20,30,40])
    print(list(s)) # <class 'pandas.core.series.Series'>
    
list_s()

7. dict(s):
To convert Series object to dictionay

dict keys --> Series obhect index labels
dict values ==> Series objects values

Example :

def dict_s():
    s = pd.Series([10,20,30,40])
    print(dict(s)) # {0: 10, 1: 20, 2: 30, 3: 40}
    
dict_s()



8. min(s)

9. max(x)
'''

def dict_s():
    s = pd.Series([10,20,30,40])
    print(dict(s)) # {0: 10, 1: 20, 2: 30, 3: 40}
    
# dict_s()
# Q. What is the main difference between pandas Series and python's dict object?
'''
in the case of series objects , dubplicate index lables (keys) are possible.
But
In the case of pyton's dict, dublicate key are not possible.

Example :

def conparision_dict_ser():
    s = pd.Series(
        data=['SUNNY','BUNNY','RUBBY','RAJJY'],
        index=[10,20,30,10]
    )
    print(s)
    print(dict(s)) # {0: 10, 1: 20, 2: 30, 3: 40}
    
conparision_dict_ser()

Output:
--------------
10    SUNNY
20    BUNNY
30    RUBBY
10    RAJJY
dtype: object
{10: 10    SUNNY
10    RAJJY
dtype: object, 20: 'BUNNY', 30: 'RUBBY'}



Whenever we are trying to convert Series object to Python's dict , if duplicate index labes are there, then
with those duplicate index labes, a Series object will be created and assign that series object to the corresponding
key in the dictionary, 
The adbantage of this approach is we are not missing any data in the conversaion from series object to dict.
'''


def conparision_dict_ser():
    s = pd.Series(
        data=['SUNNY','BUNNY','RUBBY','RAJJY'],
        index=[10,20,30,10]
    )
    print(s)
    print(dict(s)) # {0: 10, 1: 20, 2: 30, 3: 40}
    
# conparision_dict_ser()
    
# Creation of series object with the data from the csv file:
#-----------------------------------------------------------
'''
We know already the creation of series object from the list, dict, ndarray and scalar values.

We can create series object with the data from mulitple sources like csv file,excel file,json file,
html file , clipboard (whenever capy that save into session from there also we can create) etc..

Pandas library contains multiple functions for this like

pd.read_csv()
pd.read_excel()
pd.read_html()
pd.read_json()

etc..

We have to use read_csv() funtin to create series object with the data from csv file..

Read a comman -separated values (csv) file into DataFrame. ie this method returns DataFrame object by default but not Series object.

def read_data_csv():
    df = pd.read_csv('student.csv',usecols=['name'],squeeze=True)
    print('Return type of df ',type(df))
    print(df)

read_data_csv()

Note : if data contains only one column , then we can get Series object directly by passing
squeeze=True


# Create Series object where name is index label and fee as values by  using student.csv file

def read_data_csv():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     
                     squeeze=True)
    print('Return type of df ',type(df))
    print(df)

read_data_csv()


# Create a Series object from any csv file which is available online
# https://www.stats.govt.nz/large-datasets/csv-files-for-download/
#sample csv file from net
# https://www.stats.govt.nz/assets/Uploads/Annual-enterprise-survey/Annual-enterprise-survey-2021-financial-year-provisional/Download-data/annual-enterprise-survey-2021-financial-year-provisional-csv.csv

def read_data_csv():
    df = pd.read_csv('https://www.stats.govt.nz/assets/Uploads/Annual-enterprise-survey/Annual-enterprise-survey-2021-financial-year-provisional/Download-data/annual-enterprise-survey-2021-financial-year-provisional-csv.csv',
                    usecols=['Variable_name','Value'],
                    index_col='Variable_name'
                    )
    print('Return type of df ',type(df))
    s = df.squeeze()
    print('now return type ',type(s))
    print(s)
    print('Name ',s.name)
    print('Size ',s.size)

read_data_csv()

'''

# Handling missing data

'''

The importance of count () method:
---------------------------------
Series.count()
Returns the number od non-NA/null observation in the Series.

Note : in the csv file --> blank/null/NaN/nan is always treated as NaN.
But None is not treated as NaN
But from the python None is also treated as missing data

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s)
    print(s.count())# count method igonre nan
    print(s.size) # sizeis not going to ignore  nana
    # 
read_csv_file()


size attribute vs count() method
--------------------------------------
size attribute returns the number of values including NAs and null values.
But count() method returns number of non-NA/null observations in the Series.


# isnull() / isna() method:
----------------------------
Series.isnull()
    Detect missing values

Returns boolean same-sized objects indicating if the values are NA.
NA values, such as None, numpy.NaN, get mapped to True values. eveythings else gets mapped to False values.

Ecample :

    
s = pd.Series([10,20,30,None,pd.NA,np.nan])
s1 = s.isnull()
print(s1)

Output:
--------
0    False
1    False
2    False
3     True
4     True
5     True
dtype: bool


# To get only missing data :
# ---------------------------

Example :
    
s = pd.Series([10,20,30,None,pd.NA,np.nan])
s1 = s[s.isnull()]
print(s1)

Output:
--------
3    None
4    <NA>
5     NaN
dtype: object

Note : s.isnull() returns boolean series object, which is used for boolean masking to select only values where NA is available.

Another Example:
----------------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s[s.isnull()]
    print(s1)
read_csv_file()

Output:
--------------
python pd1.py
name
F   NaN
F   NaN
E   NaN
Name: fee, dtype: float64

Note : s.isna() is just alias name for s.isnull(). hence we can use these two methods interchangeably.

# How to get the number of missing values:
------------------------------------------------

There are multiple ways:
1st ways:
---------
s.size - s.count()

2nd ways:
----------
s.isnull().sum()

Note : While performing sum() operation, False is treated as 0 and True 1.


s.notnull()/s.notna()

# question
1. how to check whether SEries object contains NaN / null or not?
by using hasnans attribute

how to get only missing values? 
s[s.isnull()] or s[s.isna()], s.loc[s.isnull()]


to get number of missing values

s.size - s.count()

How to get non missing values?

s[s/notnull()]   s.loc[s.notnull()]
s[s.notna()]     s.loc[s.notna()]


how to get number of non missing values

s.notnull().sum()   s.notna().sum()
s[s.notnull()].size    s[s.notna()].size
'''

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s[s.isnull()]
    print(s1)
read_csv_file()
    
# s = pd.Series([10,20,30,None,pd.NA,np.nan])
# s1 = s[s.isnull()]
# print(s1)
    
# How to drop NAs?
#----------------------------
'''
Series class contains dropna() method for this.

Series.dropna()
    Return a new Series with missing values removed.Because of this methods there is no change in the existing Series objects.


Example

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s.dropna()
    print(s1)
read_csv_file()

Output:
---------
name
A      5000.0
B     70000.0
C     50500.0
D    500505.0
E      6666.0
X    500505.0
E      6666.0
X      6666.0
F      5000.0
X     34234.0
E     70000.0
F     70001.0
Name: fee, dtype: float64

In the above examplw s contains NAs. because dropna() method returns a new Series object.


# How to drops NAs in the existing objects only?
----------------------------------------------
We have to set inplace parameter with True value.The default value is False

Example:

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s.dropna(inplace=True)
    print(s1)# None
    print(s)
read_csv_file()

Output:

name
A      5000.0
B     70000.0
C     50500.0
D    500505.0
E      6666.0
X    500505.0
E      6666.0
X      6666.0
F      5000.0
X     34234.0
E     70000.0
F     70001.0
Name: fee, dtype: float64


#How to replace NAs with our required values?
----------------------------------------------
By using fillna(), we can replace NAs with our required value.

This method will returns a new Series object, If we want to perform modification in the existing object 
we have to use implace parameter.

Example:

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s.fillna(0)
    print(s1)
read_csv_file()

Output:
--------------
name
A      5000.0
B     70000.0
C     50500.0
D    500505.0
E      6666.0
F         0.0
X    500505.0
E      6666.0
F         0.0
X      6666.0
E         0.0
F      5000.0
X     34234.0
E     70000.0
F     70001.0
Name: fee, dtype: float64


# To perform modification in the existing only:
----------------------------------------------

Example:


def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    s1 = s.fillna(value=0,inplace=True)
    print(s1)# None
    print(s)
read_csv_file()

Output:

name
A      5000.0
B     70000.0
C     50500.0
D    500505.0
E      6666.0
F         0.0
X    500505.0
E      6666.0
F         0.0
X      6666.0
E         0.0
F      5000.0
X     34234.0
E     70000.0
F     70001.0
Name: fee, dtype: float64



'''


# Basic Staticstics for Series objects:

'''
1. s.sum() 
Returns the sum of values, present inside the Series object.

Example:

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.sum())
read_csv_file()

output:
----------
1325743.0

Note: this will ignore NAs Automatically.


2. s.mean():
------------------
Returns mean value of the series

Mean means average

Example:


def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.mean())
read_csv_file()

output:
----

110478.58333333333

Example-2:


def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.sum()/s.count())
read_csv_file()

output:
-----------------
110478.58333333333


3. s.median()
-----------------
Returns middle element in the sorted list of values.

if no. of values are odd: returns middle value

1,2,3,4,5,6,7 --> median is 4

if no. of value even : returns mean of middle 2 values.

1,2,3,4,5,6,7,8 --> median is 4.5 (mean of 4 and 5)

Example:
-------------
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.median())
read_csv_file()


4. s.var()
-----------
it returns the variance of alues of the Series object.

Example :
-------------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.var())
read_csv_file()


output:
------------
33922724088.265152


5. s.std()
-------------------
It returns of standard deviation of values of series objects.

It is the square root of variance.

Example:
-------------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.std())
read_csv_file()

output:
-------
184181.22621012476


Another Example
------------------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.std())
    op = int(s.std()**2) == int(s.var())
    print(op)# True
read_csv_file()

6. s.mode():
---------------------
Returns the  most repeated value. ie most frequently occured values

Example
-----------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.mode())
read_csv_file()


output:
-----------

0    5000.0
Name: fee, dtype: float64


# How to finds the number of times value repeated
--------------------------------------------------
By using boolean masking
s[s=5000]

Example :
---------
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.mode())
    print(s[s==5000].size)#4

read_csv_file()

output:

4

Example2:
-----------
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.mode())
    print(s[s==s.mode()[0]].size)#4

read_csv_file()

output:
--------------
4


s.value_count()
-------------------
Series.balue_counts(normalize=False,sort=True,ascending=Fale,bins=None,dropna=True)

Return Series containing counts of unique values

The resulting object will be in descending order so that the first element is the most frequently-occuring 
element, excludes NA values by default.

Example:
-----------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.value_counts())

read_csv_file()


Output:-
------------
fee
5000.0      4
6666.0      3
70000.0     1
50500.0     1
500505.0    1
34234.0     1
70001.0     1
Name: count, dtype: int64


Note : if we use normalize=True then we will get frequency in percentace

Example:
---------------
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.value_counts(normalize=True))

read_csv_file()

Output:
---------------
5000.0      0.333333
6666.0      0.250000
70000.0     0.083333
50500.0     0.083333
500505.0    0.083333
34234.0     0.083333
70001.0     0.083333
Name: proportion, dtype: float64



s.min() and s.max()
---------------

Return minimum value present into series


Example:
-------------
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Min value',s.min())
    print('Max value ',s.max())

read_csv_file()


Output:
--------------
Min value 5000.0
Max value  500505.0


s.describe():
------------------
Generatic descripive stastics

Example:-
-------------

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Describe ',s.describe())

read_csv_file()

Outpute:
-----------
Describe  count        12.000000
mean      63769.833333
std      139955.128760
min        5000.000000
25%        5000.000000
50%        6666.000000
75%       55375.000000
max      500505.000000
Name: fee, dtype: float64

Importance of describe methond
-------------------------------
It generate descriptive stastics like count, mean, std, min, max, etc
Before analysing our data , it is recommended to  use this mothod to get descriptive statistics
about our series objects.


Note

min 63769
25% of values are less than or rqual to 5000
50 % of values are  less thar or equal to 6666, it is the median value.
75% of values are less than or equal to 55375
max 500505


=========student.csv file============
roll_no,name,fee
101,A,5000
102,B,70000
103,C,50500
104,D,500505
105,E,6666
106,F,NA
107,X,5000
108,E,6666
109,F,NA
110,X,6666
111,E,NA
112,F,5000
113,X,34234
114,E,5000
115,F,70001

=======================

# Excecise:
--------------
1. sepearate non nulls from the studens series, which is generated from students.csv file, and assign 
this series to existing_marks variable?

Answer:
---------
def seperate_non_nulls():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('ORIGINAL SERIES ',s)
    print('SIZE ',s.size)
    # print('Describe ',s.describe())
    non_nulls = s.dropna()
    print('NON Nulls ',non_nulls)
    print(non_nulls.size)

seperate_non_nulls()

Answer-2:


def not_null():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    student = df.squeeze()
    existing_marks = student[student.notnull()]
    print('existing_marks ',existing_marks)

not_null ()




2. Find the sum of all sudent marks?

Example :-
----------

def sum_of_stuend_marks():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('total student marks ',s.sum())

sum_of_stuend_marks()


3. find the students whose marks are >= 500?

Example:-
-----------
def students_marks_greter_than_eql_to_500():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Result ',s.loc[s>=500])
    print('Result ',s[s>=500])
    print('Result ',s.get(s>=500))

students_marks_greter_than_eql_to_500()




4. find the sum of all student marks which are >= 500?

Example:-
------------

def students_marks_greter_than_eql_to_500():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Result ',s.loc[s>=500].sum())
    print('Result ',s[s>=500].sum())
    print('Result ',s.get(s>=500).sum())

students_marks_greter_than_eql_to_500()


5. How Many students got marks less than 350?

Answer:
-------
def no_of_student_get_less_than_350():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Result ',s.loc[s<=350].count())
    print('Result ',s[s<=350].count())
    print('Result ',s.get(s<=350).count())

no_of_student_get_less_than_350()


6. How Many studets got marks >= 400?

Answer:
---------
def no_of_student_get_gt_400():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print('Result ',s.loc[s>=400].count())
    print('Result ',s[s>=400].count())
    print('Result ',s.get(s>=400).count())

no_of_student_get_gt_400()

7. find highest marks in the series
s.max()
8. field list marks in the series
s.min()

'''

# finding index labels associated with mas value and min value:
# --------------------------------------------------------------
'''
Without using readymade methods:


def max_and_min_marks():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.max())
    print(s[s==s.max()].index)
    print(s[s==s.max()].index[0])

max_and_min_marks()

Output:
-----------
800.0
Index(['E'], dtype='object', name='name')
E

# We can do same thing directly byusing ready made methods:
idxmax() and idmin()

idxmax()
It returns index label associated with mas value


def max_and_min_marks():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.idxmax())
    print(s.idxmin())
max_and_min_marks()

s.idxmax() ==> Returns index label associated with max value.
s.idxmin() ==> Returns index label associated with min value..
s.max()    ==> Returns only max value but not index label
s.min()    ==> Returns only min value but not index label


Note :  If multiple max/min values then idxmax and idxmin returns only first matched index label.
'''

# Finding first n largest and smallest values : nlargest() and nsmallest():
#--------------------------------------------------------------------------
'''
s.nlargest(n=5) ==> Returns the largest n elements

Example:-
--------------


def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.nlargest(n=3))
comman()

OUTPUT:-
----------

name
E    800.0
E    500.0
E    500.0
Name: fee, dtype: float64




s.nsmallest(n=5) ==> Returns the smallest n elements

Example:-
---------

def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.nsmallest(n=3))
comman()

OUTPUT:
-----------
name
F    100.0
X    150.0
X    200.0
Name: fee, dtype: float64

'''
# Sorting the values by using short_values() method:
#------------------------------------------------
'''
This method is helpful to sort only values byt not index labels.

Syntax:-
=========
Series.sort_values(axis=0,ascending=True,inplace=False,kind='quicksort'
na_position='last')

This method is always going to return new series objects.

PARAMETERS:
==============
ascending: If True, sort values in ascending order, otherwise descending . 
Default is True Which is ment for ascending order.

inplace: default False
If True, perform operation in -place.


If we are not passing this parameter then sort_values() method
return a new series object.

If we wants to sort in exisiting object only then we have to set inplace=True

kind :choice of sorting algoritham
{'quicksort', 'mergesort', 'heapsort', stable'}, default ' quicksort'


na_position : {'first' or 'last' } default 'last'

Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at the end.


Example :
---------

def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.sort_values())
comman()


def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    abc=s.copy()
    new = abc.sort_values(inplace=True)
    print(abc)
    print(new)
comman()


# sorting based on index labels by using sort_index() method:
Exactly same as sort_values() method including parameter except that sorting based on 
index labels but not based on values.

Note : sort_values() : Returns a new Series object sorted based on values.
sort_index() : returns a new Series object sorted based on index labels.

'''

# BASIC ARITHMETIC OPERATION FOR SERIES OBJECTS:
#----------------------------------------------------
'''
1. Scalar means contant value.
-------------------------------
scalar means constant value.

We can perform arithmetic operation betweeen series object and scalar value.
Operation will be performed for every element.

eg:
----
s = pd.Series([10,20,30,40,50])
print(s)
print(s+10)
print(s-3)
print(s*3)
print(s/2)

Note : If value the NA, then after performing scalar operation the result 
is always NA Only.

Reason:
---------
If we perform any operation on NA then result is always NA.

Example
------------
s = pd.Series([10,20,30,40,50,pd.NA])
print(s)
print(s+10)
print(s-3)
print(s*3)
print(s/2)


2. Arithmetic operation between 2 series objects:
--------------------------------------------------
We can perform arithmetic opeation between two Series objects.
These opeartion will be performed only on matched indexex.
For unmatched indexes, NaN will be returned. 

Example:
-----------

s1 = pd.Series([10,20,30,4])
s2 = pd.Series([10,20,30,4])
print(s1)
print(s1+s2)

Output:
0    10
1    20
2    30
3     4
dtype: int64
0    20
1    40
2    60
3     8
dtype: int64

Example-2:
-----------
s1 = pd.Series([10,20,30,4],index=['A','B','C','D'])
s2 = pd.Series([10,20,30,4],index=['C','D','E','F'])
print(s1)
print(s1+s2)

Output:
---------
A     NaN
B     NaN
C    40.0
D    24.0
E     NaN
F     NaN
dtype: float64

Note:- Series class contains equivalent methods for arithmetic opeaation

s1+s2  ==> s1.add(s2)
s1-s2  ==> s1.sub(s2)
s1*s2  ==> s1.mul(s2)
s1/s2  ==> s1.div(s2)


Example:-


s1 = pd.Series([10,20,30,4],index=['A','B','C','D'])
s2 = pd.Series([10,20,30,4],index=['C','D','E','F'])
print(s1)
print(s1.add(s2))

Output:-
----------
A     NaN
B     NaN
C    40.0
D    24.0
E     NaN
F     NaN
dtype: float64

fill_value parameter:
--------------------
We can pass fill_value parameter for add() , sub(), mul() and div() methods.
IF the matched index is not available, then fill_value will be considered.

Example:
---------


s1 = pd.Series([10,20,30,4],index=['A','B','C','D'])
s2 = pd.Series([10,20,30,4],index=['C','D','E','F'])
print(s1)
print(s1.add(s2,fill_value=0))

Output:
------------
name
F   NaN
F   NaN
E   NaN
Name: fee, dtype: float64
A    10
B    20
C    30
D     4
dtype: int64
A    10.0
B    20.0
C    40.0
D    24.0
E    30.0
F     4.0


Example:-

s1 = pd.Series([10,np.NaN],index=['A','Z'])
s2 = pd.Series([10,20],index=['A','B'])
print(s1)
print(s1.add(s2,fill_value=0))

OUTPUT:-
---------

A    10.0
Z     NaN
dtype: float64
A    20.0
B    20.0
Z     NaN
dtype: float64


fill_value parameter is the advantage of add()  methods when compare with 
+ operator.



Cumlative Operations / Progressive Operations:
-------------------------------------------
There are multiple cumulative operations applicable for Series Object.

s.cumsum():

Example:-
-------------

def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.sum())
    print(s.cumsum())
comman()

OUTPUT:-
-------------
4130.0
name
A     300.0
B     700.0
C     930.0
D    1180.0
E    1680.0
F       NaN
X    1880.0
E    2680.0
F       NaN
X    2980.0
E       NaN
F    3080.0
X    3230.0
E    3730.0
F    4130.0
Name: fee, dtype: float64

Note : Bydefault cumsum() method ignores NAs while performing cumulative sum 
operations. By using skipna parameter we can customize this behaviour, The 
default value is True

Example:-


def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.sum())
    print(s.cumsum(skipna=False))
comman()

Output:-
-------
4130.0
name
A     300.0
B     700.0
C     930.0
D    1180.0
E    1680.0
F       NaN
X       NaN
E       NaN
F       NaN
X       NaN
E       NaN
F       NaN
X       NaN
E       NaN
F       NaN
Name: fee, dtype: float64

Note :-  any number with NaN is always NaN


s.prod() and s.cumprod():
---------------------------------
s.prod() :- Returns the product of all values.
s.cumprod()  :-  Returns the comulative products of values


Example:-
--------------

def comman():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.prod())
    print(s.cumprod())
comman()

OUTPUT:-
-----------
4.968e+29
name
A    3.000000e+02
B    1.200000e+05
C    2.760000e+07
D    6.900000e+09
E    3.450000e+12
F             NaN
X    6.900000e+14
E    5.520000e+17
F             NaN
X    1.656000e+20
E             NaN
F    1.656000e+22
X    2.484000e+24
E    1.242000e+27
F    4.968000e+29
Name: fee, dtype: float64


# s.min () and s.cummin():
-------------------------
s.min() ==> REturns the minimum values .
s.cummin() -->  Returns the cumulative minimum value including current values

Example:-
def comman():
    df = pd.Series(data=[1,2,3,4,5])
    # convert df into series object
    s = df.squeeze()
    print(s.min())
    print(s.cummin())
comman()

output:-
-------------
1

0    1
1    1
2    1
3    1
4    1
dtype: int64


# s.max() and s.cummax()
--------------------------

Example:

def comman():
    df = pd.Series(data=[1,2,3,4,5])
    # convert df into series object
    s = df.squeeze()
    print(s.max())
    print(s.cummax())
comman()


Output:-
----------------
5
0    1
1    2
2    3
3    4
4    5
dtype: int64


# Finding Discreate Differenct by using diff() method
------------------------------------------------------
Series.diff(period=1)
    First discreate differenct of element.

Calculates the differenct of a Series element compared with another
element in Series (default is element in previous row).

Example :
---------------
def comman():
    df = pd.Series(data=[10,20,30,40,50])
    # convert df into series object
    s = df.squeeze()
    print(s.diff()) # s.duff(periods=1)
comman()

Output:-
--------------
0     NaN
1    10.0
2    10.0
3    10.0
4    10.0
dtype: float64


Note :- 
i1  ----> v1   
i2  ----> v2
i3  ----> v3
i4  ----> v4 
i5  ----> v5

periods=1
i1  ----> v1-NaN
i2  ----> v2-v1
i3  ----> v3-v2
i4  ----> v4 -v3
i5  ----> v5-v4

periods=1
i1  ----> v1-NaN
i2  ----> v2-NaN
i3  ----> v3-v1
i4  ----> v4 -v2
i5  ----> v5-v3

Example
------------
def comman():
    df = pd.Series(data=[10,20,30,40,50])
    # convert df into series object
    s = df.squeeze()
    print(s.diff(periods=2)) # s.duff(periods=1)
comman()

OUTPUT
-----------
0     NaN
1     NaN
2    20.0
3    20.0
4    20.0
dtype: float64


periods = -1 : (difference with next element)
------------------------------------------------
i1  ----> v1-v2
i2  ----> v2-v3
i3  ----> v3-v4
i4  ----> v4 -v5
i5  ----> v5-NaN


Example
----------------
def comman():
    df = pd.Series(data=[10,20,30,40,50])
    # convert df into series object
    s = df.squeeze()
    print(s.diff(periods=-1)) # s.duff(periods=1)
comman()

OUTPUT:-
--------------
0   -10.0
1   -10.0
2   -10.0
3   -10.0
4     NaN
dtype: float64

Note :- This diff() method is very helpful while working with Time 
Series in DataScience.

# Filtering element of Series based on values:
---------------------------------------------------
By Using boolean masking , we can filter elements.

eg:- to Filter all students whose marks are less than 300

Answer: -
-----------
def lt300(x):
    return x < 300
print('callable ')
def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s[s<300]) # Boolean Masking
    print(s.loc[s<300]) # Boolean Masking
    print(s[lt300])# passing callable object
read_csv_file()

Output:-

callable 
name
C    230.0
D    250.0
X    200.0
F    100.0
X    150.0
Name: fee, dtype: float64
name
C    230.0
D    250.0
X    200.0
F    100.0
X    150.0
Name: fee, dtype: float64
name
C    230.0
D    250.0
X    200.0
F    100.0
X    150.0
Name: fee, dtype: float64


Note :-
--------------
In the above example filtering  happend based on values
but not based on index labels.

    1. filter()
    2. where()
    3. mask()

If we wants to filter elements based on index labels then we should go for
filter() method:


All these method filter elements based on index labels but not based on values.


Filtering elements of Series  based on index label by using filter() methods:-
------------------------------------------------------------------------------
Syntax:-
------------
Series.filter(items=None,like=None,regex=None,exis=None)
    Subset the Series rows according to the specified index labels.

Note that this routine does not filter a series on it content,
The filter is applied to the labels of the index.

regex  parameter
------------------
eg :- Telest rows of series where index labels are starts with 'E':
We have to use regex parameter( regex - Regular Expression) 

Example :- 


def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.filter(regex='^E'))
read_csv_file()

eg :- Telest rows of series where index labels are ends with 'E':

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.filter(regex='N$'))
read_csv_file()

output:-

Series([], Name: fee, dtype: float64)

note :- need to learn about Regular Expression


like parameter
------------------

eg : To select rows of Series where index labes contain substring 'nn'

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.filter(like='nn'))
read_csv_file()

output

name
Sunny    400.0
Bunny    250.0
Hunny    200.0
Name: fee, dtype: float64


Note :- like  parameter is similar to like keyword in SQL Query

# Replacing element of Series by using where () method:
--------------------------------------------------------

Syntax:-
------------------
Series.where(cond, other=nan, inplace-False, axis=None, lebel=None,
                errors='raise',try_cast=NoDefault.no_default
                )

    Replace values where the condition is False, ie if condition is True then
    replacement wont' be happend.

    By using other parameter we can provide new value.


eg-1:
------
Replace value as 'Failed' where marks are < 300?

Example:-

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.where(lambda x: x>300,other='Failed'))
read_csv_file()

Output:-
----------
name
A        Failed
Sunny     400.0
C        Failed
Bunny    Failed
E         500.0
F        Failed
Hunny    Failed
E         800.0
F        Failed
X        Failed
E        Failed
F        Failed
X        Failed
E         500.0
F         400.0
Name: fee, dtype: object


# Replacing element of Series by using mask() method:-
---------------------------------------------------------
Replpace values where condition is True.
Example:-


def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.where(lambda x: x<300,other='Failed'))
read_csv_file()

Output:-
------------

name
A        Failed
Sunny    Failed
C         230.0
Bunny     250.0
E        Failed
F        Failed
Hunny     200.0
E        Failed
F        Failed
X        Failed
E        Failed
F         100.0
X         150.0
E        Failed
F        Failed
Name: fee, dtype: object


Example2: Replace value as first class where marks in
between > 500

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.where(lambda x: x<500,other='First class'))
read_csv_file()


Output:-

name
A              300.0
Sunny          400.0
C              230.0
Bunny          250.0
E        First class
F        First class
Hunny          200.0
E        First class
F        First class
X              300.0
E        First class
F              100.0
X              150.0
E        First class
F              400.0
Name: fee, dtype: object

NOTE:-  mask() replaces value if the condition is True where as where() replaces
values if the condition False.

TRANSFORMING SERIES OBJECT:-
-----------------------------
Transforming means updating values of Series object.

There are 2 types of transformations
    1. Sport Transformation 
    2. Global Transformation


1. Sport Transformation:-
-------------------------------
A subset of records will be updated but not all.
We can perform this operation by using update() method.

2. Global Transformation:-
-----------------------------
It will update full set of records/all records

We can perform this operation by using either map() method or apply() methods.



Transforming Series object by  using update () method:-
--------------------------------------------------------



'''

def read_csv_file():
    df = pd.read_csv('student.csv',
                     usecols=['name','fee'],
                     index_col='name',
                     )
    # convert df into series object
    s = df.squeeze()
    print(s.where(lambda x: x<500,other='First class'))
read_csv_file()

