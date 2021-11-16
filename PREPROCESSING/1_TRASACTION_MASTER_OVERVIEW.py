# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # OVERVIEW OF THE TRANSACTION MASTER TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC STEPS
# MAGIC 
# MAGIC 1. Read data
# MAGIC 2. Count total entries
# MAGIC 3. Count total card_id
# MAGIC 4. Count total cat_code
# MAGIC 5. Count total products by each card_id
# MAGIC   - Dist/Box plot the results
# MAGIC   - Drop card_id with less than X%
# MAGIC 6. Count total products by each cat_code
# MAGIC   - Dist/Box plot the results
# MAGIC   - Drop card_id with more than X%
# MAGIC 7. Save table 

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

spark.sql('USE 10_plsa')

df = spark.sql('SELECT * FROM transactions_master')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## OVERVIEW

# COMMAND ----------

# MAGIC %md
# MAGIC Total number of entries

# COMMAND ----------

(df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Total number of customers

# COMMAND ----------

df.select('card_id').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Total number of categories

# COMMAND ----------

df.select('cat_code').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CARD ID OVERVIEW

# COMMAND ----------

df_card_cat = (df
               .groupBy('card_id')
               .agg(F.count('cat_code').alias('no_cat'))
              )

display(df_card_cat)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CAT OVERVIEW

# COMMAND ----------

df_cat_card = (df.groupBy('cat_code').agg(F.count('card_id').alias('no_card')))

display(df_cat_card)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## JOIN WITH CAT MASTERS

# COMMAND ----------

dfcm = spark.sql('SELECT * FROM category_master')

# COMMAND ----------

df_card_words = df.join(dfcm, on='cat_code', how='inner')

# COMMAND ----------

df_cw = df_card_words.groupBy('card_id','word_code').agg(F.sum('qty').alias('qty'))

# COMMAND ----------

display(df_cw)

# COMMAND ----------

spark.sql('DROP TABLE IF EXISTS tran_word_counts')

df_cw.write.saveAsTable('tran_word_counts', partitionBy='word_code')

# COMMAND ----------


