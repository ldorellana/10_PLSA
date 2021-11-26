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

# MAGIC %md
# MAGIC 
# MAGIC ## READ DATA

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/10_PLSA/DATA/ANALYSIS/CAT_V2/categories_v2.csv', header=True)


# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/CAT_V2/'
file_name = 'en_jp_categories_v2.csv'

df_codes = spark.read.csv(file_location+file_name, header=True)

# COMMAND ----------

display(df_codes)

# COMMAND ----------

df = (df
        .join(df_codes, on='word_code')
        .withColumnRenamed('word_code', 'word_name')
        .withColumnRenamed('code', 'word_code')
       )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CHECK DUPLICATES

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
# MAGIC ## DELETE UNEEDED CATEGORIES

# COMMAND ----------

df.filter(F.col('word_code') == 'N/A').count()

# COMMAND ----------

df_clean = df.filter(F.col('word_code') != 'N/A')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## RESULTS OVERVIEW

# COMMAND ----------

display(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SAVE TABLE

# COMMAND ----------

# spark.sql('USE 10_plsa')
# spark.sql('DROP TABLE IF EXISTS category_master_v2')
# df_clean.write.saveAsTable('category_master_v2')

# COMMAND ----------


