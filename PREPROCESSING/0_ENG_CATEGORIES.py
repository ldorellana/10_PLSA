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

display(df_en)

# COMMAND ----------

df_en.write.saveAsTable(name='words_jpen', mode='overwrite')

# COMMAND ----------


