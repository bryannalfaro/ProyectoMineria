# %%
import random
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
import numpy as np
import sklearn.cluster as cluster
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


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
# Filtar data
df = df[df['tohite'] < 7]
df = df[df['añoreg'] > 1750]
df = df[(df['libras'] < 9) & (df['libras'] > 4)]
df = df[df['onzas'] < 20]
df = df[(df['edadp'] < 51) & (df['edadp'] > 5)]
df = df[(df['edadm'] < 47) & (df['edadm'] > 5)]
df = df[df['tohinm'] < 4]
df = df[df['tohivi'] < 7]
df.loc[df['añoreg'] == '9', 'añoreg'] = 2009.0
df.loc[df['añoreg'] == '10', 'añoreg'] = 2010.0

# %%
# CONVIERTO TODOS LOS DATOS EN NUEMROS
# df[quantitative_vars] = df[quantitative_vars].apply(pd.to_numeric)
for column in quantitative_vars:
    df[column] = df[column].astype(float)


# %%
'''
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
pd.crosstab(index=df['gretnp'], columns=df['gretnm'], margins=True)'''

# %%
# # Haga un agrupamiento (clustering) e interprete los resultados

'''cuantitatives = df[['libras', 'onzas']]
cuantitatives.head()

cuantitatives= (cuantitatives-cuantitatives.min())/(cuantitatives.max()-cuantitatives.min())
data = np.array(cuantitatives.dropna())
data
scale = sklearn.preprocessing.scale(data)
scale'''

'''numeroClusters = range(1, 11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(scale)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.title("Gráfico de Codo")
plt.show()'''


'''clusters=  cluster.KMeans(n_clusters=3, max_iter=300) #Creacion del modelo
clusters.fit(cuantitatives) #Aplicacion del modelo de cluster

cuantitatives['cluster'] = clusters.labels_ #Asignacion de los clusters
df['cluster kmeans'] = clusters.labels_
print(cuantitatives.head())

pca = PCA(2)
pca_movies = pca.fit_transform(cuantitatives)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, cuantitatives[['cluster']]], axis = 1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Clusters de peliculas', fontsize = 20)

color_theme = np.array(['red', 'green', 'blue', 'yellow','black'])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, s = 50, c = color_theme[pca_clust_movies.cluster.fillna(0).astype(int)])

plt.show()

print(df[df['cluster kmeans'] == 0].describe())
print(df[df['cluster kmeans'] == 1].describe())
print(df[df['cluster kmeans'] == 2].describe())'''

'''# %%
kmeans = cluster.KMeans(n_clusters=3, max_iter=300)
kmeans.fit(scale)
df['cluster kmeans'] = kmeans.labels_
print(df[df['cluster kmeans'] == 0].describe())
print(df[df['cluster kmeans'] == 1].describe())
print(df[df['cluster kmeans'] == 2].describe())
kmeans_result = kmeans.predict(scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    index = np.where(kmeans_result == kmeans_cluster)
    plt.scatter(scale[index, 0], scale[index, 1])
# plt.axis([0, 10, 0, 4])
plt.show()'''

# %%
'''gaussian_model = GaussianMixture(n_components=3)
gaussian_model.fit(scale)
df['cluster gaussian'] = gaussian_model.predict(scale)
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

# %%'''

# %%
random.seed(255)
df_tree = df[['deprep', 'gretnp', 'deprem', 'gretnm', 'tohivi','tohite']].copy()

# %%
df_tree2 = df_tree.iloc[0:1000000].copy()

# %%
#Mupocu, gretnm, gretnp, areag, deprep, muprep, deprem, muprem, asisrec,pueblopm, pueblopp
nonnumeric = ['gretnm', 'gretnp', 'deprep', 'deprem']

# %%
removeNum = ['gretnm', 'gretnp', 'deprep',
             'deprem']  # quitando pueblopp, pueblopm
df_tree2 = pd.DataFrame(df_tree2)
# %%
df_tree2 = df_tree2.applymap(lambda x: np.nan if x == 0.0 else x)
# %%
# Verificando que se eliminen numeros y solo queden strings
for variableg in removeNum:
    print(df_tree2[variableg].value_counts())
    print("\n")

# %%
df_tree2 = pd.DataFrame(df_tree2)
print('INITIAL \n', df_tree2.head(20))

# Eliminando valores Nan
for i in nonnumeric:
    df_tree2.dropna(subset=[i], inplace=True)

for i in nonnumeric:
    df_tree2[i] = df_tree2[i].astype("category").cat.codes

print('FINAL \n', df_tree2.head(20))

# %%
print(df_tree2.head(20))
df_tree2.fillna(0, inplace=True)  # Llenando valores Nan de cantidades
y = df_tree2.pop('tohite')
x = df_tree2
print('SHAPING\n')
print(x.shape, y.shape)
print(x.head(5))
# %%
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x, y, test_size=0.3, train_size=0.7, random_state=0)

Dt_model_reg = tree.DecisionTreeRegressor(random_state=0, max_leaf_nodes=25)


Dt_model_reg.fit(x_train_reg, y_train_reg)

y_pred_reg = Dt_model_reg.predict(X=x_test_reg)
y_pred_train_reg = Dt_model_reg.predict(X=x_train_reg)

rmse = metrics.mean_squared_error(
    y_true=y_test_reg,
    y_pred=y_pred_reg,
    squared=False
)

rmse_train = metrics.mean_squared_error(
    y_true=y_train_reg,
    y_pred=y_pred_train_reg,
    squared=False
)
mse = metrics.mean_squared_error(
    y_true=y_test_reg,
    y_pred=y_pred_reg,
    squared=True
)

mse_train = metrics.mean_squared_error(
    y_true=y_train_reg,
    y_pred=y_pred_train_reg,
    squared=True
)
print("-----------------------------------")
print(f"El error (rmse) de test es: {rmse}")
print("-----------------------------------")

print("-----------------------------------")
print(f"El error (rmse) de train es: {rmse_train}")
print("-----------------------------------")
print("-----------------------------------")
print(f"El error (mse) de test es: {mse}")
print("-----------------------------------")

print("-----------------------------------")
print(f"El error (mse) de train es: {mse_train}")
print("-----------------------------------")
plt.figure(figsize=(23, 10))
tree.plot_tree(Dt_model_reg, feature_names=df_tree2.columns,
               fontsize=7, filled=True, rounded=True)
# plt.show()
# %%
# cm = confusion_matrix(y_test_reg, y_pred_reg)
# print('Confusion matrix \n', cm)
# graf = sns.heatmap(cm, annot=True, cmap='Blues')
# graf.set_title('Matriz de confusion\n\n')
# graf.set_xlabel('\nPredicted Values')
# graf.set_ylabel('Actual Values ')
# plt.show()

# # %%
# accuracy = accuracy_score(y_test_reg, y_pred_reg)
# precision = precision_score(y_test_reg, y_pred_reg, average='micro')
# recall = recall_score(y_test_reg, y_pred_reg, average='micro')
# f1 = f1_score(y_test_reg, y_pred_reg, average='micro')
# print('Accuracy: ', accuracy)
# print('Precision: ', precision)
# %%
y = df_tree2.iloc[:,4].values
x = df_tree2
LE1 = LabelEncoder()

x["deprep"] = np.asarray(LE1.fit_transform(x["deprep"])).astype('float32').reshape((-1,1))
x["gretnp"] = np.asarray(LE1.fit_transform(x["gretnp"])).astype('float32').reshape((-1,1))
x["deprem"] = np.asarray(LE1.fit_transform(x["deprem"])).astype('float32').reshape((-1,1))
x["gretnm"] = np.asarray(LE1.fit_transform(x["gretnm"])).astype('float32').reshape((-1,1))

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

ann.fit(X_train,Y_train,batch_size=32,epochs = 10)

accuracy = ann.evaluate(X_test, Y_test, verbose=0)[1]
print('Accuracy: %.2f' % (accuracy*100))
