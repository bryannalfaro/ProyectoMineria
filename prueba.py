# %%
import numpy as np
import pandas as pd
fruit_list = [('Orange', '0.0', 'Yes'),
              ('Mango', '2.0', 'No'),
              ('kiwi', 0.0, 'No'),
              ('Apple', 'Raul', 'Yes'),
              ('Pineapple', 64, 'No'),
              ('Kiwis', 84, 'Yes')]

# Create a DataFrame object
df = pd.DataFrame(fruit_list, columns=['Name', 'Price', 'Stock'])

df2 = df.copy()

var = ['Price', 'Name']
for var1 in var:
    for i in range(len(df2)):
        try:
            if float(df2.iloc[i][var1]) == 0.0:
                df2.at[i, var1] = None
                print("changed", df2.iloc[i][var1])

        except:
            pass

    df2 = df2.dropna(subset=[str(var1)])


print("fff", df2)
# %%
