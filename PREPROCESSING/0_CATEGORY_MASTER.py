# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CATEGORY MASTER REVIEW

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To do:
# MAGIC 1. Drop unneded rows
# MAGIC 2. Drop uneeded columns
# MAGIC 3. Drop null values
# MAGIC 4. Rename columns
# MAGIC 5. Clear duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## IMPORT LIBRARIES

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/10_PLSA/DATA/ANALYSIS/category_final.csv', header=True)

# COMMAND ----------

display(df)

# COMMAND ----------

# check if there is any null values

print(f'Rows: {df.count()} | Cols: {len(df.columns)}')

df = df.dropna()

print(f'Rows: {df.count()} | Cols: {len(df.columns)}')

# COMMAND ----------

# check for duplicates

display(df.groupBy('cat_code').agg(F.count('word_code').alias('word_code')).filter('word_code > 1'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## RESULTS OVERVIEW

# COMMAND ----------

df_final = df

# COMMAND ----------

print(df_final.count())
display(df_final.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SAVE TABLE

# COMMAND ----------

# spark.sql('USE 10_plsa')
# spark.sql('DROP TABLE IF EXISTS category_master')
# df_final.write.saveAsTable('category_master')
