# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CREATION OF DOCUMENTS FOR LDA

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Each customer represents a document for LDA classification
# MAGIC Preprocesssing Algorithm:
# MAGIC 1. Keep only card_id shoppers
# MAGIC 2. Keep only card_id with at least 10 visits
# MAGIC 
# MAGIC 
# MAGIC Algorithm to create customer document:
# MAGIC 1. Get the journal data for the customers
# MAGIC 2. Group by cusotmer, [item or category (cluster category)]
# MAGIC 3. Count by category
# MAGIC 4. Convert to count vecotrizer
# MAGIC 5. Calculate the TIFID values
# MAGIC 6. Run the LDA algorithm

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

ls ../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Keep columns:  
# MAGIC card id = 19  
# MAGIC qty = 14  
# MAGIC itmcd = 8  

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/10_PLSA/DATA/ANALYSIS/yamanka_test.csv')

df = df.select('_c19', '_c8', '_c14')

# COMMAND ----------

display(df)

# COMMAND ----------

df = (df
 .withColumnRenamed('_c19', 'card_id')
 .withColumnRenamed('_c14', 'qty')
 .withColumnRenamed('_c8', 'itmcd')
)


# COMMAND ----------

cust_df = (df
      .filter('qty > 0 AND card_id <> "000000000000000000"' )
      .groupBy('card_id','itmcd')
      .agg(F.sum('qty').alias('qty'))
      .orderBy(['card_id','qty'], ascending=False)
     
     )

# COMMAND ----------

display(cust_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

pivoted = (cust_df
        .groupBy('card_id')
        .pivot('itmcd')
        .sum()
        .na.fill(0)
       )

# COMMAND ----------

inputcols = [col for col in pivoted.columns if col != 'card_id']

vectorized = (
  VectorAssembler(inputCols=inputcols, outputCol='features').transform(pivoted).select('card_id','features')
)

# COMMAND ----------

display(vectorized)



# COMMAND ----------

condensed = (cust_df
        .groupBy('card_id', 'itmcd')
        .agg(F.sum('qty'))
        .groupBy('card_id')
        .agg(F.)
        .na.fill(0)
       )
