# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/reports/11_30v2/

# COMMAND ----------

path = '/FileStore/files/LDA_MODELS/reports/11_30v2/'
file = 'PV2NO_CLUST16_11_30_purchase_data.csv'

# COMMAND ----------

df = spark.read.csv(path+file, header=True).drop('_c0')

columns = df.columns[0:6]
clusters = df.columns[6:]

# add a str at the begining of each cluster number
clusters = [f'c{clust}' for clust in clusters]
cols = columns + clusters

# rename the columns
df = df.toDF(*cols)

# create the unpvito string
unpivotCols = ' ,'.join([f'"{clust}", {clust}' for clust in clusters])

nclust = len(clusters)
unpivotExpr = f'stack({nclust}, {unpivotCols}) as (cluster_group, amount)'
df_final = (df.select(*columns, F.expr(unpivotExpr)).fillna(0))

df_pandas = df_final.toPandas()
df_pandas.to_csv(f'../../dbfs{path}db_{file}')

# COMMAND ----------


