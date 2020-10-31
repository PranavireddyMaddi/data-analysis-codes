# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For visualizing the distribution of data 


# importing data set using pandas
mba = pd.read_csv("mba.csv")
mba.head(10)
mba.describe()
mtcars = pd.read_csv("mtcars.csv")

# Skewnewss and kurtosis
mba.skew()
mba.kurt()

# Summary stats using describe function 
mtcars.describe()

# Group by function examples
# Example -1


mtcars.groupby(["carb"])["mpg"].mean() # Aggregate with respect to carb
mtcars.groupby(["gear"])["mpg","disp"].mean() 

# Example -2
mtcars.groupby(["gear","carb","vs"])["mpg"].mean() # Aggregate with respect to multiple columns
mtcars.groupby(["gear","carb"])["mpg","disp"].mean() 


# Visualizations 
# Graphical Representation of data


# Histogram
plt.hist(mba['gmat']);plt.xlabel("gmat");plt.ylabel("Freq") # left skew 
plt.hist(mba['workex']);plt.xlabel("workex");plt.ylabel("Freq")
# Multiple plots in single frame using subplot function 
j=1
for i in list(mba.columns):
    plt.subplot(1,3,j)
    plt.boxplot(mba[i])
    j=j+1
    plt.title(i)
    

#Boxplot
plt.boxplot(mba['gmat']);plt.ylabel("GMAT")# for vertical
plt.boxplot(mba['gmat'],1,'bo',0)# For Horizontal
help(plt.boxplot)


# Barplot
# bar plot we need height i.e value of each data
# left - for starting point of each bar plot of data on X-axis(Horizontal axis). Here data is mba['gmat']
index = np.arange(773) # np.arange(a)  = > creates consecutive numbers from 
# 0 to 772 

mba.shape # dimensions of data frame 
plt.bar(index,height = mba["gmat"]) # initializing the parameter 
# left with index values 


# Histograms for all features of mtcars in a single frame 
j = 1
for i in list(mtcars.columns):
    plt.subplot(4,3,j)
    j = j+1
    plt.hist(mtcars[i])
    plt.xlabel("values")
    plt.ylabel("Freq")
    plt.title(i)
plt.tight_layout() 

# 2 - way table 
pd.crosstab(mtcars.gear,mtcars.cyl)
mtcars.am.value_counts()
mtcars.cyl.nunique()
# Visualizing the 2 - way table in barplot format 
pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar")


pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar") # bar plot for 2 - way table
mtcars.gear.value_counts().plot(kind="pie") # pie chart 

import numpy as np
plt.plot(mtcars.drat,mtcars.mpg,"ro") # scatter plot of single variable

plt.plot(np.arange(32),mtcars.mpg,"ro-")
plt.plot(np.arange(len(mtcars.mpg[mtcars.gear==3])),mtcars.mpg[mtcars.gear==3],"ro-")

help(plt.plot) # explore different visualizations among the scatter plot
mtcars.mpg.groupby(mtcars.gear).plot(kind="line")
# Scatter plot between different inputs

plt.plot(mtcars.mpg,mtcars["hp"],"ro");plt.xlabel("mpg");plt.ylabel("hp")

mtcars.hp.corr(mtcars.mpg)
mtcars.corr()
# ro  indicates r - red , o - points 

# group by function 
mtcars.groupby(mtcars.gear).median() # summing up all mpg with respect to gear
mtcars.gear.value_counts()
mtcars.cyl.value_counts()
# pie chart
mtcars.gear.value_counts().plot(kind="pie")

mtcars.mpg.groupby(mtcars.gear).plot(kind="line")
#mtcars.gear.plot(kind="pie")
# bar plot for count of each category for gear 
mtcars.gear.value_counts().plot(kind="bar")

pd.crosstab(mtcars.gear,mtcars.carb).plot(kind="bar")
# histogram of mpg for each category of gears 
mtcars.mpg.groupby(mtcars.gear).plot(kind="hist") 
mtcars.mpg.groupby(mtcars.gear).count()

# line plot for mpg column
mtcars.mpg.plot(kind='area') 
plt.plot(np.arange(32),mtcars.mpg,"ro")

# Data type conversion 
# float to string
mtcars.mpg.astype(str)
mtcars.gear.astype(str)
mtcars.groupby(mtcars.gear).count()

# lambda usage example 
mtcars.groupby("gear").apply(lambda x: x.mean())
mtcars.groupby(mtcars.gear).mean()
mtcars.apply(lambda x: x.mean())

# Representing the count of each category in pie and bar chart visualization 
mtcars.gear.value_counts().plot(kind="pie")
mtcars.gear.value_counts().plot(kind="bar")
mtcars.head()
pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar",stacked=True,grid=True)
plt.scatter(mtcars.mpg,mtcars.wt)
mtcars.plot(kind="scatter",x="mpg",y="wt")
mtcars.mpg.plot(kind="hist")

import seaborn as sns
# getting boxplot of mpg with respect to each category of gears 
sns.boxplot(x="gear",y="mpg",data=mtcars)

# Scatter plot between 2 features and color based on gear type
sns.relplot(x="mpg",y="hp",hue="gear",data=mtcars,kind="scatter")

# histogram of each column and 
# scatter plot of each variable with respect to other columns 
sns.pairplot(mtcars.iloc[:,0:4])

sns.kdeplot(np.arange(mtcars.shape[0]),mtcars.mpg,shade=True)
# Density plot between 2 features
sns.kdeplot(mtcars.disp,mtcars.mpg,cmap="Reds", shade=True);sns.scatterplot(mtcars.disp,mtcars.mpg)

# Density plot for univariate (single feature)
sns.distplot(mtcars.mpg) # make hist=False to exclude histogram in the density plot 

# Multiple plots in a single frame - Density plot 
j=1
for i in list(mtcars.columns):
    plt.subplot(3,4,j)
    sns.distplot(mtcars[i])
    j=j+1

# Scatter plot between "mpg" and "wt" with respect to categorical feature "carb" using FacetGrid
# hue --> takes the categorical feature name 
# inplace of hue we can also use row and col to get the visualizations in separate graphs but in a single
# frame 
sns.FacetGrid(mtcars,hue="carb").map(plt.scatter,"mpg","wt").add_legend()
sns.FacetGrid(mtcars,hue="gear").map(plt.scatter,"mpg","disp").add_legend()


# Boxplot for mpg for each category of cylinder 
sns.boxplot(x="cyl",y="mpg",data=mtcars)
sns.FacetGrid(mtcars,row="cyl").map(sns.kdeplot,"mpg").add_legend() # inplace of row="cyl" try using hue="cyl" and col="cyl"
# or with different categorical feature to get different visualization 
            

# All possible scatter plots and histogram for each feature
sns.pairplot(mtcars[["mpg","hp","disp","wt"]])

# Scatter plot and density plot with color based on category of gear 
sns.pairplot(data = mtcars,vars=["mpg","hp","wt"],hue="gear",kind="scatter")

# distplot for continuous feature with respect to different categories of a categorical feature
sns.distplot(mtcars.wt[mtcars.gear==3],rug=True,hist=False,label= "three")
sns.distplot(mtcars.wt[mtcars.gear==4],rug=True,hist=False,label="four")
sns.distplot(mtcars.wt[mtcars.gear==5],rug=True,hist=False,label="five")
sns.plt.show()
sns.plt.legend()

# Mixed plots - giving color to scatter plot based on gear category and different axes as per 
# cyl category 
sns.FacetGrid(data=mtcars,hue="gear",row="cyl").map(plt.scatter,"mpg","disp").add_legend()

# some misc plots
sns.catplot(x="gear",y="mpg",hue="cyl",data=mtcars)
sns.catplot(x="carb",y="mpg",hue="cyl",data=mtcars,col="gear")

# Boxplot for mpg for each category of cylinder and color given based on gear category 
sns.factorplot("cyl","mpg","gear",data=mtcars,kind="box")


# Heatmaps visualizations
# Example -1
temp_data = pd.pivot_table(mtcars,values="mpg",index="gear",columns="cyl",aggfunc="mean",fill_value=0)
sns.heatmap(temp_data,annot=True,linewidths=0.5) 

# Example -2
temp_data = pd.pivot_table(mtcars,values ="mpg",index="gear",columns="carb",aggfunc="mean",fill_value=0)
sns.heatmap(temp_data,annot=True,linewidths=0.4)

# Example -3
temp_data = pd.pivot_table(mtcars,values ="mpg",index="cyl",columns="carb",aggfunc="mean",fill_value=0)
sns.heatmap(temp_data,annot=True,linewidths=0.4,cmap="YlGnBu")

