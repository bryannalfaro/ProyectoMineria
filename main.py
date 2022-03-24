# %%
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import numpy as np
import sklearn.cluster as cluster
import sklearn.preprocessing
from sklearn.mixture import GaussianMixture

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

# %%
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
               df9, df10], axis=0, ignore_index=True)
# df.to_csv('datosBase.csv')

# %%
print(df.shape)
print(df.columns.to_list())

# %%
# Tabla de frecuencias variables cualitativas
cualitative_vars = ['gretnm', 'escivm', 'depnam', 'mupnam', 'naciom', 'ocupam', 'asisrec', 'sitioocu', 'escolap', 'escolam', 'paisnacm', 'paisnacp', 'paisrem', 'paisrep', 'pueblopm', 'pueblopp', 'tipoins',
                    'viapar', 'ciuopad', 'depnap', 'mupnap', 'naciop', 'ocupap', 'deprem', 'muprem', 'depreg', 'mupreg', 'mesreg', 'depocu', 'mupocu', 'areag', 'mesocu', 'sexo', 'tipar', 'deprep', 'muprep', 'gretnp', 'escivp', 'ciuomad']
print(len(cualitative_vars))
for var in cualitative_vars:
    data = df[var].value_counts()
    print(data)
    plt.figure(figsize=(15, 5))
    sns.barplot(data.index, data.values, alpha=0.8)
    plt.title(f'Frecuencia de datos cualitativos para {var}')
    plt.ylabel('Cantidad')
    plt.xlabel(var)
    plt.xticks(rotation='vertical')
    plt.show()

# %%
quantitative_vars = ['añoreg', 'libras', 'onzas', 'diaocu',
                     'edadp', 'edadm', 'tohite', 'tohinm', 'tohivi', 'añoocu']

# %%
# CAMBIAR LA EL DATO 'Ignorado' POR 0.0
df.replace('Ignorado', 0.0, inplace=True)

# %%
# CONVIERTO TODOS LOS DATOS EN NUEMROS
# df[quantitative_vars] = df[quantitative_vars].apply(pd.to_numeric)
for column in quantitative_vars:
    df[column] = df[column].astype(float)

# %%
for var in quantitative_vars:
    print("\n===== Evaluacion de normalidad de la variable ",
          var, ' ===== \n', df[var])
    data = df[var]
    plt.hist(data, color='green')
    plt.title(f'Histograma para {var}')
    plt.xlabel(var)
    plt.ylabel('Cantidad')
    plt.show()
    qqplot(data, line='s')
    plt.title(f'QQplot para {var}')
    plt.show()

    print('Curtosis: ', stats.kurtosis(data))
    print('Asimetria: ', stats.skew(data))

# %%
pd.crosstab(index=df['gretnm'], columns=df['tohite'], margins=True)
pd.crosstab(index=df['gretnp'], columns=df['tohite'], margins=True)
pd.crosstab(index=df['deprep'], columns=df['deprem'], margins=True)
pd.crosstab(index=pd.crosstab(index=df['deprep'], columns=df['deprem'], margins=True), columns=df['tohite'], margins=True)
pd.crosstab(index=df['gretnp'], columns=df['gretnm'], margins=True)
# # Haga un agrupamiento (clustering) e interprete los resultados

cuantitatives = df[['libras', 'onzas']]
cuantitatives.head()
data = np.array(cuantitatives.dropna())
data
scale = sklearn.preprocessing.scale(data)
scale

numeroClusters = range(1, 11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(scale)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.title("Gráfico de Codo")
plt.show()

# %%
kmeans = cluster.KMeans(n_clusters=3, max_iter=300)
kmeans.fit(scale)
kmeans_result = kmeans.predict(scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    index = np.where(kmeans_result == kmeans_cluster)
    plt.scatter(scale[index, 0], scale[index, 1])
# plt.axis([0, 10, 0, 4])
plt.show()

# %%
gaussian_model = GaussianMixture(n_components=3)
gaussian_model.fit(scale)
gaussian_result = gaussian_model.predict(scale)
gaussian_clusters = np.unique(gaussian_result)

# graficar los grupos de Mezcla Gaussiana
for gaussian_cluster in gaussian_clusters:
    index = np.where(gaussian_result == gaussian_cluster)
    plt.scatter(scale[index, 0], scale[index, 1])

# mostrar el gráfico de Mezcla Gaussiana
plt.show()

# %%
