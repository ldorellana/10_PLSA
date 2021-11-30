# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CREATE ENGLISH CATEGORIES

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/CAT_V2/'
file_name = 'en_jp_categories_v21.csv'

df = spark.read.csv(file_location+file_name, header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CORRECTED NAMES

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

display(df)

# COMMAND ----------

df_en = (df
         .withColumnRenamed('word_code', 'word_en')
         .withColumnRenamed('code', 'word_code')
        )

# COMMAND ----------

from pyspark.sql import functions as F

df_lang = df_en.withColumn('word_both', F.concat_ws('<br>', 'word_code', 'word_jp', 'word_en'))

display(df_lang)

# COMMAND ----------

# spark.sql('USE 10_plsa')
# spark.sql('DROP TABLE IF EXISTS words_jpen_v2')
# df_lang.write.saveAsTable(name='words_jpen_v2', mode='overwrite')

# COMMAND ----------


