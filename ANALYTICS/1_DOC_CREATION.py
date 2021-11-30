# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CREATE A VECTORIZED VERSION OF THE DATA

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

date = '11_29'

# COMMAND ----------

spark.sql('USE 10_plsa')
count_df = spark.sql('SELECT * FROM tran_word_master ORDER BY card_id')

# COMMAND ----------

docs = (count_df
        .withColumn('sentence', F.expr('array_repeat(word_code, qty)')) # create an array of repeted words
        .groupBy('card_id') # group by card_id
        .agg(F.count('word_code').alias('unique_words'), # get the number of unique items per customer
             F.sum('qty').alias('qty'), # total number of purchases
             F.flatten(F.collect_list('sentence')).alias('doc') # convert to 1 list of words
            )
        .withColumn('qty', F.col('qty').astype('int')) # conver types
        .withColumn('unique_words', F.col('unique_words').astype('int'))
       )

# COMMAND ----------

spark.sql('DROP TABLE IF EXISTS card_id_docs')
docs.write.saveAsTable('card_id_docs')
