# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # LDA FOR 1 YEAR TRANSACTIONAL DATA
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

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## READ COUNT DATA

# COMMAND ----------

spark.sql('USE 10_plsa')
count_df = spark.sql('SELECT * FROM tran_word_counts ORDER BY card_id')

# COMMAND ----------

# MAGIC %md
# MAGIC Keep only customers that:
# MAGIC   - Have at least 15 types of items

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## FILTER MIN UNIQUE ITEMS

# COMMAND ----------

word_by_card = (count_df.groupBy('card_id')
                .agg(F.count('word_code').alias('words_no'))
               )

filtered_cust = word_by_card.filter('words_no > 20')

# COMMAND ----------

total_cust = word_by_card.orderBy('words_no').count()
cust_15ormore =filtered_cust.count()

print(f'Total customers: {total_cust}')
print(f'Cusotmers with at least 15 items: {cust_15ormore}')
print(f'Customers with less than 15 products in the year: {total_cust - cust_15ormore}')

# COMMAND ----------

count_filt = count_df.join(filtered_cust.select('card_id'), how='inner', on='card_id')

# COMMAND ----------

# MAGIC %md 
# MAGIC # TRAINING DATA 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CONVERTO TO VECTOR

# COMMAND ----------

# save the names of the unique words
words = count_filt.select('word_code').distinct().collect()
words = [row['word_code'] for row in words]

# COMMAND ----------

df_pivoted = (count_filt
              .groupBy('card_id')
              .pivot('word_code')
              .agg(F.sum('qty'))
              .fillna(0)
             )

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import LDA
from collections import defaultdict
from pyspark.ml import Pipeline

# COMMAND ----------

# Estimator
vectAssembler = VectorAssembler(inputCols=words, outputCol='features')

# Transformer
vectorized_df = vectAssembler.transform(df_pivoted).select('card_id','features')

# COMMAND ----------

# Estimator
lda_estimator = LDA(featuresCol='features', 
                    maxIter=20, 
                    seed=20, 
                    k=20, 
                    optimizer='online',
                    topicDistributionCol='topic_dist',
                    docConcentration=[0.1],
                    topicConcentration=0.1,
                   )

# Model
lda_model = lda_estimator.fit(vectorized_df)

# Transformer
docs_df = lda_model.transform(vectorized_df)

# Topics
topics = lda_model.describeTopics(maxTermsPerTopic=35)

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType

def term_word(termIndices):
  terms =[ words[index] for index in termIndices]
  return terms

term_wordUDF = F.udf(term_word, returnType=ArrayType(StringType()))

topics = (topics.withColumn('terms', term_wordUDF(F.col('termIndices'))))

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

fig = px.treemap(df_topics, path=[px.Constant("TOPICS"), 'topic', 'word_jp'], values='termWeights', )
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# COMMAND ----------



# COMMAND ----------

import plotly.express as px

fig = px.treemap(df_topics, path=[px.Constant("TOPICS"), 'topic', 'word_en'], values='termWeights', )
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# COMMAND ----------

import plotly.express as px

fig = px.treemap(df_topics, path=[px.Constant("TOPICS"), 'topic', 'word_en'], values='termWeights', )
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()
