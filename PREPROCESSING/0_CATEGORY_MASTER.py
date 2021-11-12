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

df = pd.read_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/category_beta.csv', skiprows=2)
df = spark.createDataFrame(df)

# COMMAND ----------

df_clean = (df
        .drop('No', '大分類ｺｰﾄﾞ', '大分類名称', '中分類ｺｰﾄﾞ', '中分類名称', '商品ｺｰﾄﾞ','商品名','会員買上金額', '小分類名称')
        .withColumnRenamed('小分類ｺｰﾄﾞ', 'cat_code')
        .withColumnRenamed('CATAEGORY　CODE', 'word_code')
        .withColumnRenamed('CATEGORY　NAME', 'word_name')
        .dropna()
     )

df_clean.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ELIMINATE DUPLIATES

# COMMAND ----------

df_final = (df_clean
        .groupBy('cat_code', 'word_code') 
        .agg(F.first('word_name').alias('word_name'), 
             F.count('word_name').alias('n_items')) # get the number of items per cat-word
        .orderBy('cat_code','n_items', ascending=False) # get the cat with most items first
        .groupBy('cat_code') # group by cat
        .agg(F.first('word_code').alias('word_code'), 
             F.first('word_name').alias('word_name')) # keep the word with most items per category
       )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## RESULTS OVERVIEW

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
# df_clean.write.saveAsTable('category_master')

# COMMAND ----------

spark.sql('USE 10_plsa')
cat_master = spark.sql('SELECT * FROM category_master')

# COMMAND ----------

cat_master.write.csv('/FileStore/files/10_PLSA/DATA/ANALYSIS/z', )

# COMMAND ----------

ls ../../dbfs/FileStore/files/10_PLSA/DATA/ANALYSIS/z

# COMMAND ----------


