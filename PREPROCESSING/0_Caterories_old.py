# Databricks notebook source
import pandas as pd

# COMMAND ----------

spark.sql('USE 10_plsa')

# COMMAND ----------

df = pd.read_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/old_categories.csv')

# COMMAND ----------

df_spark = spark.createDataFrame(df).withColumnRenamed('小分類ｺｰﾄﾞ', 'cat_code')

# COMMAND ----------

display(df_spark)

# COMMAND ----------

spark.sql('DROP TABLE IF EXISTS old_categories')
df_spark.write.saveAsTable('old_categories')

# COMMAND ----------


