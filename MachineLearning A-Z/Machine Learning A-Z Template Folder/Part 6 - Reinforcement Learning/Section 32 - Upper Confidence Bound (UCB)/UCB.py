# Reinforecment Learning - Upper COnfidnece Bound

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Importing DataSet
Ads_DataSet = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB

N = 10000
d = 10

ads_selected = []
Number_of_Selections = [0] * d
Sum_of_Rewards = [0] * d
Total_Reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,10):
        if Number_of_Selections[i] > 0:
            average_reward = Sum_of_Rewards[i] / Number_of_Selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / Number_of_Selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    Number_of_Selections[ad] = Number_of_Selections[ad] + 1
    Sum_of_Rewards[ad] = Sum_of_Rewards[ad] + Ads_DataSet.iloc[n, ad]
    Total_Reward = Total_Reward + Ads_DataSet.iloc[n, ad]

#Visualising the results
plt.hist(ads_selected)
plt.title("Ad Selections")
plt.xlabel("Ads")
plt.ylabel("Number of Selections")
plt.show()
