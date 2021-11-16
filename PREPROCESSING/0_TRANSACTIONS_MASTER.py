# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # PREPROCESSING FOR CUSTOMER PURCHASES

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Files contain monthly data
# MAGIC 4 columns
# MAGIC 
# MAGIC 1. Index (not needed)
# MAGIC 2. card_id
# MAGIC 3. general_category
# MAGIC 4. qty
# MAGIC 
# MAGIC Preprocesssing
# MAGIC 
# MAGIC 1. Read data
# MAGIC 2. Drop index
# MAGIC 3. Delete card_id == 0
# MAGIC 4. Check for duplicates
# MAGIC 5. Check for negatives and null
# MAGIC 6. Group by card_id.cat_code sum the values
# MAGIC 7. Save the data

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## READ DATA

# COMMAND ----------

files = ['FRESTA 20 NOV.csv',
'FRESTA 20 OCT.csv',
'FRESTA 21 APR.csv',
'FRESTA 21 AUG.csv',
'FRESTA 21 FEB.csv',
'FRESTA 21 JAN.csv',
'FRESTA 21 JUL.csv',
'FRESTA 21 JUN.csv',
'FRESTA 21 MAR.csv',
'FRESTA 21 MAY.csv',
'FRESTA 21 SEP.csv',
'FRESTA 20 DEC.csv']

# COMMAND ----------

schema = StructType([StructField('No',StringType(),True),
                         StructField('card_id',StringType(),True),
                         StructField('cat_code',StringType(),True),
                         StructField('qty',IntegerType(),True)]
                   )

main_df = spark.createDataFrame([], schema=schema)

# COMMAND ----------

for file_name in files:
  file = f'/FileStore/tables/10_PLSA/DATA/ANALYSIS/CJ/unzipped/{file_name}'
  df_new = spark.read.csv(path=file, header=True, encoding='shift_jis', schema=schema)

  main_df = main_df.union(df_new)
  
display(main_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CLEAN DATA

# COMMAND ----------

df_clean = (main_df.drop('No') # drop unneeded column
 .filter(('card_id != "0"') or ('qty > 0')) # filter unwanted values
 .dropna() # drop null values
 .withColumn('cat_code', F.regexp_replace('cat_code', '\|', '_')) # change the word_code pattern
)

print(df_clean.count(), len(df_clean.columns))

display(df_clean.head(2))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## GROUP CUSTOMERS

# COMMAND ----------

df_grouped =(df_clean
             .groupBy('card_id', 'cat_code')
             .agg(F.sum('qty').alias('qty'))
             )

# COMMAND ----------

display(df_grouped)

# COMMAND ----------

(df_grouped.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SAVE TABLE

# COMMAND ----------

spark.sql('USE 10_plsa')
spark.sql('DROP TABLE IF EXISTS transactions_master')

df_grouped.write.saveAsTable('transactions_master', partitionBy='cat_code')

# COMMAND ----------


