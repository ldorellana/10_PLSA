# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CREATE A VECTORIZED VERSION OF THE DATA

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

date = '11_24'

# COMMAND ----------

spark.sql('USE 10_plsa')
count_df = spark.sql('SELECT * FROM tran_word_master ORDER BY card_id')

# COMMAND ----------

docs = (count_df
        .withColumn('sentence', F.expr('array_repeat(word_code, qty)'))
        .groupBy('card_id')
        .agg(F.count('word_code').alias('unique_words'),
             F.sum('qty').alias('qty'),
             F.flatten(F.collect_list('sentence')).alias('doc')
            )
        .withColumn('qty', F.col('qty').astype('int'))
        .withColumn('unique_words', F.col('unique_words').astype('int'))
       )

# COMMAND ----------

spark.sql('DROP TABLE IF EXISTS card_id_docs')

docs.write.saveAsTable('card_id_docs')

# COMMAND ----------


