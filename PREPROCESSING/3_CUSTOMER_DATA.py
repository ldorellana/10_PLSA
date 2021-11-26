# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # PREPROCESSING OF CUSTOMER DATA

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/10_PLSA/DATA/ANALYSIS/FRESTA_CUSTOMER_DATA.csv', encoding='shift_jis', header=True)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

w = Window.partitionBy('card_id')

df_clean = (df
            #.select('Card ID', 'Ganeration', 'Sex')
            .withColumnRenamed('Card ID', 'card_id')
            .withColumnRenamed('Ganeration', 'age')
            .withColumnRenamed('Store Name', 'store')
            .withColumnRenamed('Sex', 'gender')
            .withColumn('Purchase SKU amount', F.col('Purchase SKU amount').astype('int'))
            .filter(F.col('card_id') != '0')
            .withColumn('maxVisits', F.max('Purchase SKU amount').over(w))
            .where(F.col('maxVisits') == F.col('Purchase SKU amount'))
            .select('card_id', 'age', 'gender', 'store')
           )

# COMMAND ----------

df.count() - df_clean.count()

# COMMAND ----------

df_clean.count()

# COMMAND ----------

display(df_clean)

# COMMAND ----------

spark.sql('USE 10_plsa')
spark.sql('DROP TABLE IF EXISTS customer_master')
df_clean.write.saveAsTable('customer_master')
