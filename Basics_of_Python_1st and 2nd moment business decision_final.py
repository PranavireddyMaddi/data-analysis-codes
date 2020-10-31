# Python Basics 

# Creating, Accessing, Modifying String data type and functions
x = "Awesome"
type(x) # Will let us know the type of data type 
len(x) # gets the count of letters including space as separate element 
x.lower() # lower() is an attribute of string data type which we use it as function to convert everything into lower
x.capitalize() 
"thisis".capitalize()
x.upper()

# Accessing elements of a string using their index values
x[1]
x[-6:]

# String conversion ==> using str() 
str(9)
str(213)

# Adding strings using +
"this "+"is "+"awesome"

# String manipulation 
# Modifying or replacing letters/ group of letters 
x.replace("e","t")
import re
re.sub("[A-Za-z' ']"," ","ada 1231zx></#$%$%#$dq #@$@zxfsd312") # Replaces the value with new values
re.sub("[^A-Za-z0-9]"," ","ASDA Sada 1231zxdq #@$@zxfsd312")
re.sub("[emA]","y",x) # it replaces all the letters e m A with new letter
x.replace("e","FF")
x.endswith("e") 

# Checks if A or B or E are present then it replaces it with " " (blank space)
re.sub("[ABE]"," ","ACBDEETWERW")

# Strings does not support assignment operator (if in case we want to replace a letter using index value)
x[0]="B" # string does not support assignment operator
# Explore more string functions 

# List data structure ==> functions, accessing, modifying elements and mathematical operations on lists
x = [1,2,3,4,5,611,100,100,12,1231,101,5,-1]
type(x)

x.insert(1,1000) # insert function will place new value at respective index position without replacing original elements
# instead it will push forward the existing elements
x.sort(reverse=True) # by default reverse ==> paramter is set to False --> Ascending order

x.reverse() # to get the Ascending order 

# Adding new elements at the end using append and extend function
x.append(["E","Y","E"])
x.append("ASDDASDAS")
x.extend("XYZ")
x.append([100,200,400,500])
x.extend("123")
x = [1,2,3,4]
y=["A","B","C"]

# Combining elements of 2 different lists into a single list by using +
x+y

# list comprehension method on list examples 
x = [1,2,1,2,4,5,7,2,1,3,5,7,8,3,4,7,8,9,0]
[i for i in x if i%2==0] # Getting all the even numbers 

# Conditional statements using => if, elif,else keywords -> examples 

# Example - 1
if (3>4):
    print ("yes")
else:
    print ("No")


# Example -2 
num = 5 
if num < 3:
    print ("A")
elif num <5:
    print ("B")
elif num < 6:
    print ("C")
elif num < 7:
    print ("D")
else:
    print ("None")

# multiple conditional statements using "and" "or" keywords 
    
if (12%3==0) and (12%4==0):
    print ("12 is a multiple of both 3 and 4")
else:
    print ("12 is a multiple of either 3 or 4")


# usage of range function 

list(range(100)) # will get you 100 consecutive values from 0 to 99
list(range(1,10)) # will get you numbers from 1 to 10 
list(range(1,10,2)) # will get you the numbers from 1 to 10 with an increment of 2

# Exmaple -1 
x = list(range(20))
for i in x:
    if (i%4==0) and (not i%3==0):
        print (i,"multiple of 4")
    elif (i%3==0) and (not i%4==0):
        print (i,"multiple of 3")
    elif (i%4==0) and (i%3==0):
        print (i,"Multiple of 3 and 4")
    else:
        print (i,"Not a multiple of either 3 or 4")


# Example -2
for i in range(10,100,2):
    print (i)

# Inputting data through console (dynamically during execution)

a = []
for i in range(1,10):
    a.append(int(input())) # input() function -> generally we use it to input data through console 


# while loop syntax 
i = 1
while i<10:
    i = i+1
    print (i)


x = [[1,2],[1,2,4],[43],[54],[2,3,4]]

# using break operation to terminate any for loop in middle
for i in x:
    if len(i)>2:
        break # comes out of the loop if the number of elements present inside list of list is more than 2
    else:
        print (i)
    
    
# Creating a user defined function - syntax 
    
# Example -1
def sum(i):
    add = 0
    for k in i:
        add = add+k
    print (add)
    
    
sum([1,2,3,5])
sum([1,2,3,4,56,7])
sum([1,2,3,4,5,6,100])


# Example -2
def MEAN(i): # taking only one parameter 
    a = sum(i)  
    b =len(i)
    print (a/b)

x = [1,2,3,4,5,6,8]
MEAN(x)  


import math

# Example -3
def confirm_prime(i):
    j=2
    if i < 2:
        print ("IDK")
    elif i == 2:
        print ("Yes")
    else:
        j=2
        a = int(math.sqrt(i))
        while (j<a):
            if i%j==0:
                #print ("No")
                break
            else:
                j=j+1
    if j >= int(math.sqrt(i)):
        print("Yes")
    else: print ("NO")


# Enter only whole numbers 
confirm_prime(100)
confirm_prime(120)
confirm_prime(7)

k = [2,3,5,7,11,13,1,100,143,12,17,19,20]
len(k)

for i in k:
    confirm_prime(i)

# Dictionary - creating, modifying, accessing elements of Dictionary
x = {"A":10,"C":-12,"B":30,"E":5,"F":120,"I":6,"G":12,"H":100,"A":100}
type(x)
x.keys()
x.values()
# Creating new elements in dictionary elements 
x["H"] = 100
x["F"] = 1000
x["C"] = 2018
del x["C"] # Deletion 

# Sorting elements of Dictionary 
y = sorted(x.items(),key = lambda j:j[1]) # By default -> Ascending order
y.reverse() # Descending order

# Usage of "in" operator

# Example - 1
if "i" in "this":
    print ("yes")
    
# Example - 2
[i for i in "thisiseawsome" if i in "i"]

# Example - 3  --> join() usage
"".join([i for i in "thisiseawsome" if not i in "i"])


# Example -4 
"+".join(["this","is","awsome"])
"".join([i for i in "thisiseawsome" if i not in "i"])

# Example -5 --> split() function usage
"this is awesome".split(" ") # splits based on splitting criterion 

# Example -6 --> to clean a string/sentence
x = "Calculations are simple with Python, and expression syntax is straightforward: the operators +, -, * and / work as expected; parentheses () can be used for grouping. "
y = ["are","with","and","is","the","as","can"] # preparing a list of words which we don't want to include in above sentence
# which we call them as stopwords
x = re.sub("[^A-Za-z'']",' ',x) # Removing everything except alphabets
use_words = [i for i in x.split(" ") if (not i in y) and (len(i)>0) ]
" ".join(use_words)


# Numpy - array creation, modifying elements, functions on array 
import numpy as np
x = [10,21,3,14,15,16]
x[2:4]
x*2
x>10
y = np.array(x) # Creating 1-D numpy array 

type(y)
y.shape # (6,) it indicates it is an 1-D numpy array 
 # Accessing elements of array using index 
y>10 # applying condition on numpy array variable 
y[y>10] # Accessing elements using condition 
y**2 # Applying exponential of 2 on every element of y
y+1 # Adding a scalar
y[y%2==0]

x = [1,2,3,4,6,7]
y = [7,8,9,10,11,12]
z1 = np.array([x,y]) # combining 2 lists and converting them into an numpy array
z1[0:3]
z1[1]
np.array(y).shape
z = np.array([x,y])
z = z.reshape(4,3) # reshaping existing shape of numpy array to different shape 
z[:,1:] # Accessing elements of multi dimensional array using ":" and index 
z[2:,1:]
# Statistics on numpy array 
np.mean(z)
np.median(z)
np.mean(z[1,:])
np.mean(z[:,1])
np.std(z)
np.var(z)
np.corrcoef(z[1,:],z[3,:])

# Few functions in numpy 
np.round(1.29212,1)
temp1 = np.random.normal(10,2,5) # generates 5 numbers with mean approx -> 10 and standard deviation -> 2 (approx) 
temp2 = np.random.normal(-10,2,5)
temp3 = np.column_stack((temp1,temp2)) # Combining 2 different arrays in column wise 
type(temp1)

# Pandas - Creating, Accessing, Manipulating and few functions on Data Frames
import pandas as pd

x1 = [1, 2, 3, 4,5] # list format 
x2 = [10, 11,12,1000]  # list format 
x3 = list(range(5))

# Creating a data frame using lists 
X = pd.DataFrame(columns = ["X1","X2","X3"]) 

X["X1"] = x1 # Converting list format into pandas series format
X["X2"] = x2 # Converting list format into pandas series format # Error 
X["X3"] = x3 

# # Converting list format into pandas series format
X["X1"] = pd.Series(x1)  
X["X2"] = pd.Series(x2) 
X["X3"] = pd.Series(x3)

# Different ways of accessing elements of data frame 

# accessing columns using "." (dot) operation
X.X1
# accessing columns alternative way
X["X1"]
# Accessing multiple columns : giving column names as input in list format
X[["X1","X2"]]
# Accessing elements using ".iloc" : accessing each cell by row and column 
# index values
X.iloc[0:3,1]
X.iloc[:,2]
X.iloc[:,:] # to get entire data frame 

# using ".loc" : accessing elements using their row indexes and correspoding column names
X.loc[0:2,["X1","X2"]]
# checking the type of variable 
type(X.X1) # pandas series object

# Example -2
# Data Frame 
x = pd.DataFrame(columns=["A","B","C"]) 

# np.random.randint(a,b,c) 
# a - > starting number
# b - > Ending number
# c - > no. of numbers to be generated 
x["A"] = pd.Series(list(np.random.randint(1,100,50)))
# np.random.choice([a,b],size=c)
# a and b = > choosing elements from a or b 
# c = > number of elements to be generated choosing from a or b
x["B"] = pd.Series(list(np.random.choice([0,1,2,3,4,5],size = 50)))
x["C"] = 10 # going to fill all the rows in "C" with value 10


# Importing a .csv file from local system

import pandas as pd
help(pd.read_csv)
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("F:\\DS\\datasets\\Datasets_BA 2\\mba.csv")
mba.head(10)
type(mba) # pandas data frame
mba.columns # accessing column names 
mba.datasrno # Accessing datasrno using "." (dot) operation
mba["workex"]
mba[["datasrno","workex"]] #  accessing multiple columns 
mba.iloc[45:51,1:3] # mba.iloc[i,j] 
# i => row index values  | j => column index values
mba.loc[45:51,["workex","datasrno"]]

# Loading cars data set
mtcars = pd.read_csv("F:\\DS\\Python Codes\\Final\\Basic Statistics _ Visualizations\\mtcars.csv")
mtcars.cyl.value_counts() # to get the count of each category of categorical feature in data frame 
mtcars.carb.value_counts().plot(kind="bar") # Plotting the count of each category in bar plot format 
mtcars.vs.value_counts()
# Accessing elements using conditional input 
mtcars.gear==3 # Applying a conditional statement on gear feature (Condition evaluation )


# Example -1
temp = mtcars[(mtcars.gear==3) | (mtcars.gear==5)] # Accessing elements whose gear value can be either 3 or 4 
temp.shape
temp.gear.value_counts()
temp.carb.value_counts()


# Example -2
# and operation (&) 
mtcars_19_2_4 = mtcars[(mtcars.gear==3) & (mtcars.mpg > 19.2)] #  and operation (&)

# Example -3
# or operation (or)
mtcars_5 = mtcars[(mtcars.gear==4) | (mtcars.gear==5)]

# Example -4
# Gear 4 and 5 cars only 
mtcars_4_5 = mtcars[(mtcars.mpg>19.2) | (mtcars.gear==4) | (mtcars.gear==6)]

# Example -5
# isin operator which functions similar to that of "or" operator 
list(range(15,21,2))
mtcars_4_5 = mtcars.loc[mtcars.mpg.isin(list(range(15,21,2))),["cyl","mpg"]]

# Example -6
list(range(15,21,1)) # get the range of values 
mtcars_2_4_5 = mtcars.loc[mtcars.carb.isin([1,4]),["mpg","carb"]]


# Example -7
mtcars_15_19 = mtcars[(mtcars.mpg>15) & (mtcars.mpg<19)]


# Creating a data frame manually from lists 
new_df = pd.DataFrame(columns=["A","B","C"])
new_df["A"] = pd.Series([1,2,3,4,5,6])    
new_df["B"] = pd.Series(["A","B","C","D"])   
new_df["C"] = pd.Series([True,False,True,False])

# Dropping rows and columns 

df2=new_df.drop(["B","C"],axis=1) # Dropping columns 
# axis = 1 represnts drop the columns 
# new_df.drop(["A","B"],axis =1, inplace=True) # Dropping columns 
# inplace = True  = > action will be effective on original data 


# Dropping rows => axis =0
mba.drop([5,9,19],axis=0,inplace=True) # ,inplace=True) # Dropping rows 
# selecting specific rows using their index values in list format
#  X.index[[1,2,3,4,5]] => dropping 1,2,3,4,5 rows 

# X.drop(X.index[[5,9,19]],axis=0, inplace = True)

#X.drop(["X1","X2"],aixs=1) # Dropping columns
#X.drop(X.index[[0,2,3]],axis=0) # Dropping rows 

# Creating a data frame using dictionary object 
x = {"A":pd.Series([1,2,3,4,5,7,8,10]),"B":pd.Series(["A","B","C","D","E","F","G"]),"C":pd.Series([1,2,3,4,5,7,8])}
new_x = pd.DataFrame(x)


# Dictionary object
dict_new = {"A":pd.Series([1,2,3,4,5,7,8]),"B":pd.Series(["A","B","C","E","F","G"]),"C":pd.Series([1,2,3,4,7,8])}
pd.DataFrame(dict_new)
dict_new.keys() 
dict_new.values()
dict_new["A"] # accessing values using the key
# In any dictionary object we have unique keys and keys must not be repeated
# values can be of any size and can be repeated 

# Business moments - Statistical measures of data 
# Some statistical measurements using functions 
mtcars.mean()
mtcars.median()
mtcars.mode()
mtcars.apply(lambda x: x.mean()) # using apply function
mtcars.isnull().sum() # count of all 

# Variacne & Standard Deviation for Population
import numpy as np
np.var(mba['gmat']) # 859.70
np.std(mba['gmat']) # 29.32

# calculating the range value 
range = max(mba['gmat'])-min(mba['gmat']) # max(mba.gmat)-min(mba.gmat)
range
