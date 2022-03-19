# %%
from matplotlib.pyplot import axis
import pandas as pd

# %%
# Carga de datos
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

# %%
df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()
df3.columns = df3.columns.str.lower()
df4.columns = df4.columns.str.lower()
df5.columns = df5.columns.str.lower()
df6.columns = df6.columns.str.lower()
df7.columns = df7.columns.str.lower()
df8.columns = df8.columns.str.lower()
df9.columns = df9.columns.str.lower()
df10.columns = df10.columns.str.lower()

# %%
# ARREGLO DE DATOS
# REMPLAZO DE LA COLUMNA munpnap POR mupnap DEBIDO A QUE ES LA MISMA DESCRIPCION
df6['mupnap'] = df6.pop('munpnap')
df7['mupnap'] = df7.pop('munpnap')
df8['mupnap'] = df8.pop('munpnap')
df9['mupnap'] = df9.pop('munpnap')
df10['mupnap'] = df10.pop('munpnap')


# %%
# REMPLAZO DE LA COLUMNA grupetma POR gretnm DEBIDO A QUE ES LA MISMA DESCRIPCION
df2['gretnm'] = df2.pop('grupetma')
df3['gretnm'] = df3.pop('grupetma')
df4['gretnm'] = df4.pop('grupetma')

# %%
# REMPLAZO DE LA COLUMNA munnam POR mupnam DEBIDO A QUE ES LA MISMA DESCRIPCION
df4['mupnam'] = df4.pop('munnam')

# df1.columns.str.lower()
#df = df1.append([df2.columns.str.lower(),df3.columns.str.lower(),df4.columns.str.lower(),df5.columns.str.lower(),df6.columns.str.lower(),df7.columns.str.lower(),df8.columns.str.lower(),df9.columns.str.lower(),df10.columns.str.lower()],ignore_index=True)

# %%
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
               df9, df10], axis=0, ignore_index=True)
df.to_csv('datosBase.csv')

# %%
print(df.shape)
print(df.columns.to_list())
