# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # LDA MODELS REVIEW
# MAGIC 
# MAGIC 
# MAGIC 1. READ VECTORIZED DATA
# MAGIC 2. GET THE MODEL
# MAGIC 3. TRANSFORM DATA
# MAGIC 4. GET MAIN CLUSTER
# MAGIC 5. COUNT CUST PER CLUSTER
# MAGIC 5. DESCRIBE TOPICS
# MAGIC   - describe topics (maxTermPerTopic)
# MAGIC   - get the word_en/word_jp for each word_code in topics
# MAGIC 6. PLOT TOPICS

# COMMAND ----------

from pyspark.ml.clustering import LDA, LocalLDAModel, DistributedLDAModel
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F

import plotly.express as px
import pandas as pd
import numpy as np
import os

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## READ COUNT DATA

# COMMAND ----------

date = '11_22'

# COMMAND ----------

spark.sql('USE 10_plsa')
df_vectorized = spark.sql('SELECT * FROM vect11_22')
df_lang = spark.sql('SELECT * FROM words_jpen')
df_words = spark.sql('SELECT * FROM category_master')

# COMMAND ----------

df_lang = df_lang.withColumn('word_both', F.concat_ws('<br>','word_jp','word_en'))

# COMMAND ----------

# get the name of the word
df_words = (df_words
            .groupBy('index')
            .agg(F.first('word_name').alias('word_name'), 
                 F.first('word_code').alias('word_code'))
            .orderBy('index')
           )

word_codes = df_words.select('word_code').collect()
word_codes = [word['word_code'] for word in word_codes]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## LDA MODELS

# COMMAND ----------

#  mv ../../dbfs/FileStore/files/LDA_MODELS/11_18_AlphaBeta/[k=10]_[iter=50]_[alg=online]_[1_0]]_[a=0_5] ../../dbfs/FileStore/files/LDA_MODELS/11_18_AlphaBeta/[k=10]_[iter=50]_[alg=online]_[a=[1_0]]_[a=0_5]

# COMMAND ----------

def run_model(model_name, models_folder):
  """run the model from the path and transform the data"""
  
  
  if 'online' in model_name:
    model_jvname = (model_name+'/').replace('[','\\[').replace(']','\\]')
    lda_model = LocalLDAModel.load(models_folder+model_jvname)
  else:
    model_jvname = (model_name+'/').replace('[','\\[').replace(']','\\]')
    lda_model = DistributedLDAModel.load(models_folder+model_jvname)
  
  # transform docs and generate topics
  df_transformed = lda_model.transform(df_vectorized)
  
  # Topics
  topics = lda_model.describeTopics(maxTermsPerTopic=20)
  
  return df_transformed, topics

# COMMAND ----------

def clust_info(df_transformed):
  """ calculate clusters information"""
  
  # count the max cluster per customer
  clusterUDF = F.udf(lambda x: x.tolist().index(max(x)), IntegerType())
  
  # get the coluster of the customer
  df_transformed = (df_transformed
                    .withColumn('cluster', clusterUDF(F.col('topicDistribution'))))
  
  # calculate the size of the cluster
  cluster_size = (df_transformed.groupBy('cluster')
                  .agg(F.count('card_id').alias('size'))
                  .orderBy('size', ascending=False)
                 )
  
  return df_transformed, cluster_size

# COMMAND ----------

def word_indexer(topics):
  
  # get the word for the specified index
  term_wordUDF = (F.udf(lambda x: [word_codes[index] for index in x], 
                        returnType=ArrayType(StringType()))
                 )
  
  # perform the changes
  clusters = (topics
            .withColumnRenamed('topic', 'cluster')
            .withColumn('terms', term_wordUDF(F.col('termIndices'))))
  
  return clusters

# COMMAND ----------

def transform_clus_data(clusters_data):
 
  df_clusters = (clusters_data
          .withColumn('zipped', F.arrays_zip('terms','termWeights')) # zip together words and weights
          .withColumn('exp', F.explode('zipped')) # convert to rows
          .select('cluster','exp.terms', 'exp.termWeights') #select columns
          .withColumnRenamed('terms', 'word_code') # rename
          .join(df_lang, on='word_code') # get the words in jp and eng
          .join(cluster_size, on='cluster') # get size of cluster      
         )

  # convert to pandas
  clusters_pandas = df_clusters.toPandas()
  
  # create column to graph clust+size
  clusters_pandas['cluster_size'] = (clusters_pandas['cluster'].astype('str') + ': ' + 
                                     clusters_pandas['size'].astype('str') + '顧客') 
  
  return clusters_pandas

# COMMAND ----------

ls '../../dbfs/FileStore/files/LDA_MODELS/11_22_AlphaBeta/|k=10|_|maxIter=50|_|alg=online|_|a=|1_1||_|b=10_0|/data'

# COMMAND ----------

models_folder = f'/FileStore/files/LDA_MODELS/{date}_AlphaBeta/'
#models_list = os.listdir('../../dbfs/'+models_folder)
models_list = ['|k=10|_|maxIter=50|_|alg=online|_|a=|1_1||_|b=10_0|']


model_results = dict()

for model_name in models_list:
  
  # load the model
  df_transformed, topics = run_model(model_name, models_folder)
  
  # get the cluster for each customer and clusters size
  df_transformed, cluster_size = clust_info(df_transformed)
  
  # index the words for undestanding
  clusters_data = word_indexer(topics)
  
  # convert arrays to rows and return as pandas
  clusters_pandas = transform_clus_data(clusters_data)
  
  # cluster size for graph
  clust_size_pd = cluster_size.toPandas()  
  
  # cluster by customer
  cust_clust = df_transformed.select('card_id','cluster').toPandas()
  
  model_results[model_name] = [clusters_pandas, clust_size_pd, cust_clust]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## REPORT

# COMMAND ----------

def graph_treemap(name, results, language):
  
  fig = px.treemap(results, path=[px.Constant(name), 'cluster_size', f'word_{language}'], values='termWeights')
  fig.update_traces(root_color="lightgrey")
  fig.update_layout(margin = dict(t=20, l=25, r=25, b=15))
  
  html = fig.to_html(include_plotlyjs=False,
                          include_mathjax=False, 
                          full_html=False, 
                          default_width='90%', 
                          default_height='100%'
                         )
  
  html = html.replace('class="plotly-graph-div"', 'class="plotly-graph-div treemap_chart"')
  
  return html

# COMMAND ----------

def graph_pie(clust_size):

  fig = px.pie(clust_size, names='cluster', values='size', )
  fig.update_layout(font={'size': 16}, uniformtext_minsize=12, uniformtext_mode='hide')
  fig.update_traces(textposition='inside', textinfo='percent+value')

  html = fig.to_html(include_plotlyjs=False,
                            include_mathjax=False, 
                            full_html=False, 
                            default_width='90%', 
                            default_height='50%'
                           )
  html = html.replace('class="plotly-graph-div"', 'class="plotly-graph-div pie_chart"')
  
  return html

# COMMAND ----------

def html_head(html_file):
  
  style = (
    '#outer > div {'
      'width:100%;'
      'display: flex;'
      'justify-content: flex-start; '
      'flex-direction: column; '
      'align-items: center; '
      'align-content: center;'
    '}'
    '.plotly-graph-div.treemap_chart {'
      'height:90vh !important;'
    '}'
    '.plotly-graph-div.pie_chart {'
      'height:70vh !important;'
    '}'
    '.space {'
      'height:5vh;'
    '}'
  )

  
  html_file.write("<!DOCTYPE html>\n")
  html_file.write(f"<html><head><style>{style}</style></head><body><div id='outer'>"+"\n")

  html_file.write('<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: "local"};</script>\n'
                  '<script src="https://cdn.plot.ly/plotly-2.4.2.min.js"></script>\n')
  
  return html_file

# COMMAND ----------

lg = ['both','jp','en']

for language in lg[:1]:
  
  file_name = f'first_report_{date}_{language}'
  loc = '../../dbfs/FileStore/files/LDA_MODELS/reports/'

  html_file = open(f"{loc}{file_name}.html",'w')
  
  
  html_file = html_head(html_file)


  for clust_no, (params, results)  in enumerate(model_results.items()):

    name = f'Grouping Test No.{clust_no+1}'

    html_treemap = graph_treemap(name, results[0], language)
    html_pie = graph_pie(results[1])

    html_file.write('<div class="space"></div>''\n')
    html_file.write(f'<h1>{name}</h1>''\n')
    html_file.write(f'<p>{params}</p>''\n')

    html_file.write(html_pie+'\n')

    html_file.write(html_treemap+'\n')
    html_file.write('<div class="space"></div>''\n')

  html_file.write("\n</div></body></html>")
  html_file.close()

# COMMAND ----------

# Create a Pandas Excel writer using XlsxWriter as the engine.

writer = pd.ExcelWriter(f'grouping_{date}.xlsx', engine='xlsxwriter')

for clust_no, (params, results)  in enumerate(model_results.items()):
  
  size_sheet = f'G no.{clust_no}-SIZE'
  cust_clust = f'G no.{clust_no}-DATA'
  
  results[1].to_excel(writer, sheet_name=(size_sheet))
  results[2].to_excel(writer, sheet_name=(cust_clust))

writer.save()

# COMMAND ----------

# mv first_report19_11_21_en.xlsx ../../dbfs/FileStore/files/LDA_MODELS/reports/first_report19_11_21_en.xlsx
