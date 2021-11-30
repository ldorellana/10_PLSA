# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # DEVELOP LDA MODEL FOR CUSTOMERS DOCS

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1. Read the docs
# MAGIC 2. Delete stop words
# MAGIC 3. Create a CountVectorizer model
# MAGIC 4. Transform de data
# MAGIC 5. Create LDA model
# MAGIC 6. Fit/Transoform de model
# MAGIC 7. Review Results

# COMMAND ----------

import os

# COMMAND ----------

date = '11_29'
new_dir = f'../../dbfs/FileStore/files/LDA_MODELS/{date}v1/'
os.mkdir(new_dir)

# COMMAND ----------

os.listdir('../../dbfs/FileStore/files/LDA_MODELS/11_29v1')

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.feature import CountVectorizer, StopWordsRemover
from pyspark.ml.clustering import LDA, LDAModel, DistributedLDAModel, LocalLDAModel

# COMMAND ----------

spark.sql('USE 10_plsa')
#df_docs = spark.sql('SELECT * from card_id_docs')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## FILTER CLIENTS

# COMMAND ----------

df_docs = spark.sql('SELECT * from card_id_docs WHERE ((unique_words >= 1) AND (qty >=2))')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## FILTER STOPWORDS

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## PIPELINE

# COMMAND ----------

stop_words_remover = StopWordsRemover(inputCol='doc', outputCol='clean_doc', stopWords=[' '])

# min frequency of a product in a clients basket
minTF = 1.0
# min clients that buy a product
minDF = 1.0
# max clients that buy a product
maxDF = 9223372036854775807
vocabSize = 700

count_vectorizer = CountVectorizer(inputCol='clean_doc', 
                                   outputCol='features', 
                                   minTF=minTF, 
                                   maxDF=maxDF, 
                                   minDF=minDF, 
                                   vocabSize=vocabSize )


# params to test
alpha = [1.1]
beta = 10 
no_clusters = 15
maxIter = 50
optimizer = 'online'

# starting model
lda = LDA(featuresCol='features', seed=20, k=no_clusters, maxIter=maxIter, optimizer=optimizer)

pipeline = Pipeline(stages=[stop_words_remover, count_vectorizer, lda])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## RUNS

# COMMAND ----------

pipeline_path = f'FileStore/files/LDA_MODELS/{date}v1/'

no_clusters = [16]

for no_cluster in no_clusters:
  
  pipeline_name = F'NO_CLUST{no_cluster}'
  
  #set the new parameter
  pipeline.getStages()[2].set(lda.k, no_cluster)
  
  pipepline_model = pipeline.fit(df_docs)
  
  pipepline_model.write().overwrite().save(pipeline_path+pipeline_name)

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/11_29v1/

# COMMAND ----------


