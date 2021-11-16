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
# MAGIC 2. Keep only card_id with at least 10 items
# MAGIC 
# MAGIC 
# MAGIC Algorithm to create customer document:
# MAGIC 1. Get the transactions_master
# MAGIC 2. Get the category_master
# MAGIC 3. Merge left with category_master
# MAGIC 4. Generate SpacerVectors per card
# MAGIC 5. Develop LDA

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

spark.sql('USE 10_plsa')
df_cm = spark.sql('SELECT * FROM category_master')
df_cj = spark.sql('SELECT * FROM transactions_master ORDER BY card_id LIMIT 100000')

# COMMAND ----------

df = (df_cm.join(df_cj, on='cat_code', how='inner'))

# COMMAND ----------

# print(f'CatM: rows-> {df_cm.count()}, cols-> {len(df_cm.columns)}')
# print(f'CusJ: rows-> {df_cj.count()}, cols-> {len(df_cj.columns)}')

# COMMAND ----------

# print(f'Merge: rows-> {df_cj.count()}, cols-> {len(df_cj.columns)}')
df = df.groupBy('card_id','word_code').agg(F.sum('qty').alias('qty'))

# print(f'words: rows-> {df.count()}, cols-> {len(df.columns)}')
#display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ASSEMBLE VECTOR

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

#display(df.head(2))

# COMMAND ----------

pivoted = (df
        .groupBy('card_id')
        .pivot('word_code')
        .sum()
        .na.fill(0)
       )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### COUNT VECTOR

# COMMAND ----------

words = [col for col in pivoted.columns if col != 'card_id']

vectorizer = VectorAssembler(inputCols=words, outputCol='features')
vectorized = vectorizer.transform(pivoted).select('card_id','features')

#display(vectorized.head(2))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### IDTF

# COMMAND ----------

from pyspark.ml.feature import IDF

weighted_df = (IDF(inputCol='features', outputCol='features_w', minDocFreq=100)
               .fit(vectorized)
               .transform(vectorized)
              )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## LDA MODEL

# COMMAND ----------

from pyspark.ml.clustering import LDA
clusters = 15
lda_model  = (LDA(seed=20, 
                  k=clusters,
                  optimizer='online',
                  maxIter=80,
                  featuresCol='features_w'
                 ).fit(weighted_df))

results = lda_model.transform(weighted_df)

# COMMAND ----------

topicIndices = lda_model.describeTopics(maxTermsPerTopic=40)

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType

def term_word(termIndices):
  
  terms =[ words[index] for index in termIndices]
  return terms

term_wordUDF = F.udf(term_word, returnType=ArrayType(StringType()))

topics = (topicIndices.withColumn('terms', term_wordUDF(F.col('termIndices'))))

# COMMAND ----------

import pandas as pd


df_lang = spark.sql('SELECT * FROM words_jpen')

df_topics = (topics
        .withColumn('zipped', F.arrays_zip('terms','termWeights'))
        .withColumn('exp', F.explode('zipped'))
        .select('topic','exp.terms', 'exp.termWeights')
        .withColumnRenamed('terms', 'word_code')
        .join(df_lang, on='word_code')
        .toPandas()
       )

# COMMAND ----------

import plotly.express as px

fig = px.treemap(df_topics, path=[px.Constant("TOPICS"), 'topic', 'word_en'], values='termWeights', )
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# COMMAND ----------

import plotly.express as px


fig = px.treemap(df_topics, path=[px.Constant("TOPICS"), 'topic', 'word_jp'], 
                 values='termWeights', hover_data={'word_code':False, 'termWeights':False,'word_code':True, 'word_en':True}, 
                )
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## EXPORTING TO EXCEL

# COMMAND ----------

ls ../../dbfs/FileStore/tables/10_PLSA/DATA/RESULTS/LDA

# COMMAND ----------

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
writer.save()

# COMMAND ----------

name = f'clusters_{clusters}'

exporter = pd.ExcelWriter(path=f'{name}.xlsx', engine='xlsxwriter')

for cluster in range(0, clusters):
  
  df_topics[df_topics['topic'] == cluster].to_excel(exporter, sheet_name=f'cluster {cluster}')
  
exporter.save()

# COMMAND ----------

ls

# COMMAND ----------

mv clusters_15.xlsx '../../dbfs/FileStore/tables/10_PLSA/DATA/RESULTS/LDA/{name}.xlsx'

# COMMAND ----------

ls '../../dbfs/FileStore/tables/10_PLSA/DATA/RESULTS/LDA/'
