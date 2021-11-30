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

display(df)

# COMMAND ----------

stop_words = ['(空白)','収納代行', 'テナント店', '特販品']

# COMMAND ----------

df_words = (df
            .groupBy('word_code')
            .agg(F.first('word_name').alias('word_name'))
            .filter(~F.col('word_name').isin(stop_words))
            .orderBy('word_code')
            .withColumn('index', F.monotonically_increasing_id())
           )

# COMMAND ----------

display(df_words)

# COMMAND ----------

df_final = (df
            .select('cat_code', 'cat_name', 'word_code')
            .join(df_words, on='word_code', how='inner')
           )

# COMMAND ----------

print(df_final.count())
display(df_final.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## EDIT CATEGORIES AT REQUEST

# COMMAND ----------

cat_code = ['000033_000001_000001_000001', '000030_000001_000001_000001']

# COMMAND ----------

df_final.count()

# COMMAND ----------

df_final = (df_final.filter(~F.col('cat_code').isin(cat_code)))

# COMMAND ----------

df_final.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SAVE TABLE

# COMMAND ----------

# spark.sql('USE 10_plsa')
# spark.sql('DROP TABLE IF EXISTS category_master')
# df_final.write.saveAsTable('category_master')

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/reports/11_30v2

# COMMAND ----------


