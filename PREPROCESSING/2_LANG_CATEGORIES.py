# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CREATE ENGLISH CATEGORIES

# COMMAND ----------

spark.sql('USE 10_plsa')
dfcat = spark.sql('SELECT word_code, FIRST(word_name) FROM category_master GROUP BY word_code')

# COMMAND ----------

display(dfcat)

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = 'word_jpen.csv'

df_en = spark.read.csv(file_location+file_name, header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CORRECTED NAMES

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df_en = df_en.withColumn('word_en', F.when(F.col('word_code') == '19_5_1', 'beer').otherwise(F.col('word_en')))

# COMMAND ----------

display(df_en.filter('word_code == "19_5_1"'))

# COMMAND ----------

from pyspark.sql import functions as F


df_lang = df_en.withColumn('word_both', F.concat_ws('<br>', 'word_code', 'word_jp', 'word_en'))

display(df_lang)

# COMMAND ----------

spark.sql('USE 10_plsa')
spark.sql('DROP TABLE IF EXISTS words_jpen')
df_lang.write.saveAsTable(name='words_jpen', mode='overwrite')
