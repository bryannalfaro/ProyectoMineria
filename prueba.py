# %%
import numpy as np
import pandas as pd
fruit_list = [('Orange', '0.0', 'Yes'),
              ('Mango', '2.0', 'No'),
              ('kiwi', 0.0, 'No'),
              ('Apple', 'Raul', 'Yes'),
              ('Pineapple', 64, 'No'),
              ('Pineapple', 84, 'Yes')]

# Create a DataFrame object
df = pd.DataFrame(fruit_list, columns=['Name', 'Price', 'Stock'])

df2 = df.copy()
print(df2['Name'].astype('category').cat.codes)
var = ['Name']
for var1 in var:
        df2[var1]= df2[var1].astype("category").cat.codes


print('porto', df2.head(7))

# %%
