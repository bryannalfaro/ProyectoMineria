import numpy as np
import pandas as pd
fruit_list = [ ('Orange', 34, 'Yes' ) ,
             ('Mango', 'Juan', 'No' ) ,
             ('banana', 14, 'No' ) ,
             ('Apple', 'Raulmeco', 'Yes' ) ,
             ('Pineapple', 64, 'No') ,
             ('Kiwi', 84, 'Yes')  ]

#Create a DataFrame object
df = pd.DataFrame(fruit_list, columns =
                  ['Name' , 'Price', 'Stock'])

print(df)

df_tree = df.iloc[0:2]

print(df_tree)

'''for i in range(len(df)):
    try:
        if int(df['Price'][i]):

            print("here")
            df['Price'][i] = np.nan

    except:
        pass
df1 = df.dropna(subset=['Price'])
print(df1)'''