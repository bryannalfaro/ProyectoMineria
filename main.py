import pandas as pd

#Carga de datos
df1 = pd.read_spss('2009.sav')
df2 = pd.read_spss('2010.sav')
df3 = pd.read_spss('2011.sav')
df4 = pd.read_spss('2012.sav')
df5 = pd.read_spss('2013.sav')
df6 = pd.read_spss('2014.sav')
df7 = pd.read_spss('2015.sav')
df8 = pd.read_spss('2016.sav')
df9 = pd.read_spss('2017.sav')
df10 = pd.read_spss('2018.sav')


df = df1.append([df2,df3,df4,df5,df6,df7,df8,df9,df10],ignore_index=True)
print(df.shape())