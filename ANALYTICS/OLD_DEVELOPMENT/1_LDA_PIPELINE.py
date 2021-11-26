# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # LDA PIPELINE
# MAGIC 
# MAGIC 
# MAGIC 1. READ COUNT DATA
# MAGIC   - Switch db
# MAGIC   - Read table
# MAGIC 2. CONVERT TO VECTOR (TF)
# MAGIC   - Get unique values for word_code
# MAGIC   - Pivot table (groupby card_id, word_code -> columns)
# MAGIC   - Vectorize table (VectorAssembler with unique word_code)
# MAGIC 3. CALCULATE IDF
# MAGIC   - Pass the vectorized values to an IDF
# MAGIC 4. GENERATE THE LDA MODEL
# MAGIC   - Set the parameters (seed, maxIter, optimizer, no clusters, features_col)
# MAGIC   - Fit the model
# MAGIC   - Transform the data
# MAGIC 5. DESCRIBE TOPICS
# MAGIC   - describe topics (maxTermPerTopic)
# MAGIC   - get the word_en/word_jp for each word_code in topics
# MAGIC 6. PLOT TOPICS

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import LDA
from pyspark.sql import functions as F
from collections import defaultdict
from pyspark.ml import Pipeline

# COMMAND ----------

# date = '11_22'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## READ COUNT DATA

# COMMAND ----------

spark.sql('USE 10_plsa')
count_df = spark.sql('SELECT * FROM tran_word_master ORDER BY card_id')

# COMMAND ----------

# MAGIC %md
# MAGIC Keep only customers that:
# MAGIC   - Have at least 15 types of items

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## FILTER MIN UNIQUE ITEMS

# COMMAND ----------

min_difproducts = 20
min_products = 50

word_by_card = (count_df.groupBy('card_id')
                .agg(F.count('word_code').alias('dif_products'),
                     F.sum('word_code').alias('qty_products'))
               )

filtered_cust = word_by_card.filter(f'dif_products > {min_difproducts} or qty_products > {min_products}')

# COMMAND ----------

total_cust = word_by_card.count()
filtered =filtered_cust.count()

print(f'Total customers: {total_cust}')
print(f'Cusotmers filtered: {filtered}')
print(f'Customers not used for fitting model: {total_cust - filtered}')

# COMMAND ----------

count_filt = count_df.join(filtered_cust.select('card_id'), how='inner', on='card_id')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## PREPARE DF

# COMMAND ----------

# pivot the table

df_filt_piv = (count_filt
              .groupBy('card_id')
              .pivot('index')
              .agg(F.sum('qty'))
              .fillna(0)
             )

df_piv = (count_df
              .groupBy('card_id')
              .pivot('index')
              .agg(F.sum('qty'))
              .fillna(0)
             )

# COMMAND ----------

word_codes = df_filt_piv.columns[1:]

# Estimator
vectAssembler = VectorAssembler(inputCols=word_codes, outputCol='features')

df_filt_vect = vectAssembler.transform(df_filt_piv).select('card_id','features')
df_vect = vectAssembler.transform(df_piv).select('card_id','features')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SAVE VECTORIZED DATA

# COMMAND ----------

spark.sql(f'DROP TABLE IF EXISTS vect{date}')
df_vect.write.saveAsTable(f'vect{date}')

spark.sql(f'DROP TABLE IF EXISTS filt_vect{date}')
df_filt_vect.write.saveAsTable(f'filt_vect{date}')

# COMMAND ----------

spark.sql('USE 10_plsa')
df_filt_vect = spark.sql(f'SELECT * FROM filt_vect{date}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## LDA MODELS

# COMMAND ----------

# starting model
lda = LDA(featuresCol='features', seed=20, )

# params to test
alpha = [[1.1]]
beta = [10] 
no_clusters = list(range(10,21))
maxIter = [50]
optimizer = ['online']


# create grid
params = (ParamGridBuilder()
          .addGrid(lda.k, no_clusters)
          .addGrid(lda.maxIter, maxIter)
          .addGrid(lda.optimizer, optimizer)
          .addGrid(lda.docConcentration, alpha)
          .addGrid(lda.topicConcentration, beta)
         ).build()

len(params)

# COMMAND ----------

# MAGIC %sh ls ../../dbfs/FileStore/files/LDA_MODELS/11_22_AlphaBeta/

# COMMAND ----------

models_path = f'FileStore/files/LDA_MODELS/{date}_AlphaBeta/'
models = {}

for grid in params:
  for key,value in grid.items():
    lda.set(key, value)
    
  # create the model name
  vals = (list(grid.values()))
  params_key = ['k=', 'maxIter=', 'alg=', 'a=', 'b=']
  model_name = '_'.join([f"[{param_key+str(value).replace('.','_')}]" for param_key,value in zip(params_key, vals)])
  model_name = model_name.replace('[','|').replace(']','|')

  lda_model = lda.fit(df_filt_vect)
  
  # save the fitted model
  lda_model.write().overwrite().save(models_path+model_name)
  

# COMMAND ----------


