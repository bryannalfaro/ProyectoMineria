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
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

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
# print(len(cualitative_vars))
# for var in cualitative_vars:
#     data = df[var].value_counts()
#     print(data)
#     plt.figure(figsize=(15, 5))
#     sns.barplot(data.index, data.values, alpha=0.8)
#     plt.title(f'Frecuencia de datos cualitativos para {var}')
#     plt.ylabel('Cantidad')
#     plt.xlabel(var)
#     plt.xticks(rotation='vertical')
#     plt.show()

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
    plt.title(f'BoxPlot para {var}')
    plt.boxplot(data)
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(data.values, data.index)
    plt.title(f'Gráfica de Dispersión para {var}')
    plt.ylabel('Cantidad')
    plt.xlabel(var)
    plt.show()
    qqplot(data, line='s')
    plt.title(f'QQplot para {var}')
    plt.show()

    print('Curtosis: ', stats.kurtosis(data))
    print('Asimetria: ', stats.skew(data))

# %%
# Estadistica descriptiva
for var in quantitative_vars:
    try:
        print("\n===== Estadistica descriptiva de la variable ",
              var, ' ===== \n')
        print(df[var].describe())
    except:
        print('fallo')

# %%
# Correlaciones
corr_df = df.corr(method='pearson')

plt.figure(figsize=(15, 15))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', mask=np.triu(
    np.ones_like(corr_df, dtype=bool)), vmin=-1, vmax=1)
plt.show()

# %%
# Cruce de datos
pd.crosstab(index=df['gretnp'], columns=df['tohite'], margins=True)
# %%
pd.crosstab(index=df['gretnm'], columns=df['tohite'], margins=True)
# %%
pd.crosstab(index=df['deprep'], columns=df['deprem'], margins=True)
# %%
pd.crosstab(index=df['gretnp'], columns=df['gretnm'], margins=True)

# %%
# # Haga un agrupamiento (clustering) e interprete los resultados

cuantitatives = df[['libras', 'onzas', 'edadp', 'edadm', 'añoreg']]
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
df10.replace('Ignorado', 0.0, inplace=True)
cuantitatives2 = df10[['libras', 'onzas', 'edadp', 'edadm', 'añoreg']]
data = np.array(cuantitatives2.dropna())
data
scale = sklearn.preprocessing.scale(data)
fig, eje = plt.subplots(figsize=(1,1))
fig.set_size_inches(18, 7)

eje.set_xlim([-0.1, 1])
eje.set_ylim([0, len(scale) + (4) * 10])
print(len(scale))
clabels = kmeans.fit_predict(scale)
promedio = silhouette_score(scale, clabels)
silueta_prueba = silhouette_samples(cuantitatives2, clabels)
min =10

for i in range(0, 3):
    valores = silueta_prueba[clabels == i]
    valores.sort()
    size = valores.shape[0]
    max = min + size
    color = cm.nipy_spectral(float(i) / 3)
    eje.fill_betweenx(np.arange(min, max), 0, valores, facecolor=color, edgecolor=color, alpha=0.7,)
    eje.text(-0.05, min + 0.5 * size, str(i))
    min = max + 10

eje.set_title('Silueta')
eje.set_xlabel("Valores")
eje.set_ylabel("Nombre")
eje.axvline(x=promedio, color="red", linestyle="--")
eje.set_yticks([]) 
eje.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# %%
