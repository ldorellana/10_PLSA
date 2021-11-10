# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # PREPROCESSING DATA

# COMMAND ----------

# libraries
import pyspark.sql.functions as f
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CREATE DB

# COMMAND ----------

spark.sql('CREATE DATABASE IF NOT EXISTS 10_PLSA')
spark.sql('USE 10_PLSA')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TB - CATEGORY MASTER

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = '_clasification_master_分類マスタ.csv'
file_type = 'csv'

df = pd.read_csv('/dbfs'+file_location+file_name, encoding='cp932', dtype='str')

df.to_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/clasification_master.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Set the table schema

# COMMAND ----------

from pyspark.sql.types import (StructField, StructType, StringType,
                               IntegerType, FloatType)

data_shema = [StructField('dep_code', StringType(), True),
              StructField('dep_name', StringType(), True),
              StructField('c1_code', StringType(), True),
              StructField('c1_name', StringType(), True),
              StructField('cat_code', StringType(), True),
              StructField('cat_name', StringType(), True),
              StructField('c2_code', StringType(), True),
              StructField('c2_name', StringType(), True),
              StructField('c3_code', StringType(), True),
              StructField('c3_name', StringType(), True),
              StructField('c1_merge', StringType(), True),
              StructField('c2_merge', StringType(), True),
              StructField('c3_merge', StringType(), True),
              StructField('cat_merge', StringType(), True),
              StructField('sales', FloatType(), True),
              StructField('amount', FloatType(), True),
              StructField('target', StringType(), True),
             ]

table_schema = StructType(fields=data_shema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the data

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = 'clasification_master.csv'


# The applied options are for CSV files. For other file types, these will be ignored.
df = (spark.read.csv(path=file_location+file_name, header=True, schema=table_schema))

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Table Overview

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

 display(df.head(4))

# COMMAND ----------

# show number of unique values
display(df.agg(*(f.countDistinct(c).alias(c) for c in df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save table

# COMMAND ----------

df.write.saveAsTable(name='category_master', mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TB - ITEM MASTER

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = '_product_master_商品マスタ.csv'
file_type = 'csv'

df = pd.read_csv('/dbfs'+file_location+file_name, encoding='cp932', dtype='str')

df.to_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/product_master.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Set the table schema

# COMMAND ----------

from pyspark.sql.types import (StructField, StructType, StringType,
                               IntegerType, FloatType)

data_shema = [StructField('item_cod', StringType(), True),
              StructField('item_name', StringType(), True),
              StructField('dep_code', StringType(), True),
              StructField('dep_name', StringType(), True),
              StructField('c1_code', StringType(), True),
              StructField('c1_name', StringType(), True),
              StructField('cat_code', StringType(), True),
              StructField('cat_name', StringType(), True),
              StructField('c2_code', StringType(), True),
              StructField('c2_name', StringType(), True),
              StructField('c3_code', StringType(), True),
              StructField('c3_name', StringType(), True),
              StructField('sales', IntegerType(), True),
              StructField('amount', IntegerType(), True),
              StructField('price', IntegerType(), True),
              StructField('c1_merge', StringType(), True),
              StructField('c2_merge', StringType(), True),
              StructField('c3_merge', StringType(), True),
              StructField('cat_merge', StringType(), True),
             ]

table_schema = StructType(fields=data_shema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the data

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = 'product_master.csv'

# The applied options are for CSV files. For other file types, these will be ignored.
df = (spark.read.csv(path=file_location+file_name, header=True, schema=table_schema))

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Table Overview

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# show number of unique values
display(df.agg(*(f.countDistinct(c).alias(c) for c in df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save table

# COMMAND ----------

df.write.saveAsTable(name='item_master', mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TB - CUSTOMER MASTER

# COMMAND ----------

ls ../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = '_Customer_attribute_information_顧客属性情報.csv'
file_type = 'csv'

df = pd.read_csv('/dbfs'+file_location+file_name, encoding='cp932', dtype='str')

df.to_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/customer_master.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Set the table schema

# COMMAND ----------

from pyspark.sql.types import (StructField, StructType, StringType,
                               IntegerType, FloatType)

data_shema = [StructField('dummy_id', StringType(), True),
              StructField('card_id', StringType(), True),
              StructField('gen_code', StringType(), True),
              StructField('gen_name', StringType(), True),
              StructField('age_code', StringType(), True),
              StructField('age_name', StringType(), True),
              StructField('no_stores', StringType(), True),
              StructField('main_store', StringType(), True),
              StructField('store_name', StringType(), True),
              StructField('cust_gno', StringType(), True),
              StructField('cust_gname', StringType(), True),
              StructField('cl_no', StringType(), True),
              StructField('cl_name', StringType(), True),
             ]

table_schema = StructType(fields=data_shema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the data

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = 'product_master.csv'

# The applied options are for CSV files. For other file types, these will be ignored.
df = (spark.read.csv(path=file_location+file_name, header=True, schema=table_schema))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# show number of unique values
display(df.agg(*(f.countDistinct(c).alias(c) for c in df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save table

# COMMAND ----------

df.write.saveAsTable(name='customer_master', mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TB - JOURNAL MASTER

# COMMAND ----------

ls -l ../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Set the table schema

# COMMAND ----------

from pyspark.sql.types import (StructField, StructType, StringType,
                               IntegerType, FloatType)

data_shema = [StructField('id', IntegerType(), True),
              StructField('cat_merge', StringType(), True),
              StructField('no_purchases', FloatType(), True),
             ]

table_schema = StructType(fields=data_shema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the data

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/10_PLSA/DATA/ANALYSIS/'
file_name = '修正PLSA用データ161024.csv'

# The applied options are for CSV files. For other file types, these will be ignored.
df = (spark.read.csv(path=file_location+file_name, header=True, schema=table_schema))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# show number of unique values
display(df.agg(*(f.countDistinct(c).alias(c) for c in df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save table

# COMMAND ----------

df.write.saveAsTable(name='transactions_master', mode='overwrite')
