# implementing Apriori algorithm from mlxtend

# conda install -c conda-forge mlxtend

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


groceries = []
# As the file is in transaction data we will be reading data directly 
with open("D:\\CQ\\Python\\datasets\\groceries.csv","r") as f:
    groceries = f.read()



# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))       


# Creating Data Frame for the transactions data 
import pandas as pd
# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
np.mean(np.array(X))

frequent_itemsets = apriori(X,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(list(range(1,11)),height = frequent_itemsets.support[0:10],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape


rules.head(20)
rules.sort_values('confidence',ascending = False,inplace=True)












 
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


x=[7,4,9,12,12,7,6,4,9]

print(to_list(x))

"tirtle"+"ninja"

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

set(tuple(x))


unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


