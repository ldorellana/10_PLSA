# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # OVERVIEW OF THE LDA MODEL

# COMMAND ----------

date = '11_30'
model_date = '11_29'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## libraries

# COMMAND ----------

from pyspark.ml.clustering import LDA, LocalLDAModel, DistributedLDAModel
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## READ DATA

# COMMAND ----------

spark.sql('USE 10_plsa')

df_words = spark.sql('SELECT * FROM category_master')
df_lang = spark.sql('SELECT * FROM words_jpen')
df_docs = spark.sql('SELECT * FROM card_id_docs')
df_cust = spark.sql('SELECT * FROM customer_master')
df_transactions = spark.sql('SELECT * FROM transactions_master')
df_old_cat = spark.sql('SELECT * FROM old_categories')

# COMMAND ----------

df_cat_master = (df_old_cat
                 .join(df_words['word_code', 'cat_code', 'word_name'], on='cat_code')
                 .withColumnRenamed('word_code','clust_category')
                 .withColumnRenamed('word_name','clust_category_name')
                )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## RUN PIPELINE

# COMMAND ----------

def run_pipeline(pipeline_name, pipeline_folder, df_vectorized):
  """run the model from the path and transform the data"""
  
  
  pipeline_model = PipelineModel.load(pipeline_folder+pipeline_name)
  
  # transform docs and generate topics
  df_transformed = pipeline_model.transform(df_vectorized)
  
  # Topics
  vocabulary = pipeline_model.stages[1].vocabulary
  topics = pipeline_model.stages[2].describeTopics(maxTermsPerTopic=50)
  
  return df_transformed, topics, vocabulary

# COMMAND ----------

def clust_info(df_transformed):
  """ calculate clusters information"""
  
  # count the max cluster per customer
  clusterUDF = F.udf(lambda x: x.tolist().index(max(x)), IntegerType())
  
  # get the coluster of the customer
  df_transformed = (df_transformed
                    .withColumn('cluster', clusterUDF(F.col('topicDistribution'))))
  
  # calculate the size of the cluster

  total_cust = df_transformed.count()

  cluster_size = (df_transformed
                  .groupBy('cluster')
                  .agg(F.count('card_id').alias('size'),
                       F.round(100*(F.count('card_id')/total_cust), 2).alias('percent'),
                      )
                  .orderBy('size', ascending=False)
                 )

  df_trans_cust = df_transformed.join(df_cust, on='card_id')

  # get gender rate
  gender_rate = (df_trans_cust
                 .groupby('cluster')
                 .pivot('gender')
                 .agg(F.count('card_id')).alias('gender')
                 .fillna(0)
                )

  # get age rate
  age_rate = (df_trans_cust
                 .groupby('cluster')
                 .pivot('age')
                 .agg(F.count('card_id')).alias('age')
                 .fillna(0)
                )

  cluster_summary = (cluster_size
                     .join(gender_rate, on='cluster', how='inner')
                     .join(age_rate, on='cluster', how='inner')
                    )
  
  return df_transformed, cluster_summary

# COMMAND ----------

def word_indexer(topics, vocabulary):
  
  # get the word for the specified index
  term_wordUDF = (F.udf(lambda x: [vocabulary[index] for index in x], 
                        returnType=ArrayType(StringType()))
                 )
  
  # perform the changes
  clusters = (topics
            .withColumnRenamed('topic', 'cluster')
            .withColumn('terms', term_wordUDF(F.col('termIndices'))))
  
  return clusters

# COMMAND ----------

def transform_clus_data(clusters_data, df_lang, cluster_size):
 
  df_clusters = (clusters_data
          .withColumn('zipped', F.arrays_zip('terms','termWeights')) # zip together words and weights
          .withColumn('exp', F.explode('zipped')) # convert to rows
          .select('cluster','exp.terms', 'exp.termWeights') #select columns
          .withColumnRenamed('terms', 'word_code') # rename
          .join(df_lang, on='word_code') # get the words in jp and eng
          .join(cluster_size.select('cluster','size', 'percent'), on='cluster') # get size of cluster
          .withColumn('cluster_info', F.concat_ws('- ', 'cluster', 'percent'))
          .withColumn('cluster_info', F.concat_ws('%, 人数', 'cluster_info', 'size'))
         )
 
  return df_clusters

# COMMAND ----------

def items_clust(cust_clust, df_transactions, df_old_cat, df_words): 
  
  clust_tran = (cust_clust.select('card_id','cluster')
                .join(df_transactions['card_id','cat_code','qty'], on='card_id', how='inner')
                .groupBy('cat_code')
                .pivot('cluster')
                .agg(F.sum('qty').alias('qty'))
                .fillna(0)
                .join(df_cat_master, on='cat_code', how='inner')
               )

  category_columns = ['cat_code', '部門名称', '中分類名称', '小分類名称', 'clust_category', 'clust_category_name']
  cluster_columns = set(clust_tran.columns).difference(set(category_columns))
  columns = category_columns + sorted([int(col) for col in cluster_columns])
  columns = [str(col) for col in columns]

  clust_tran = clust_tran.select(*columns).orderBy('cat_code')
  
  return clust_tran

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/11_29v1/

# COMMAND ----------

pipeline_folder = f'/FileStore/files/LDA_MODELS/{model_date}v1/'
os.listdir('../../dbfs'+pipeline_folder)

# COMMAND ----------

pipeline_list = ['NO_CLUST16'] #os.listdir('../../dbfs/'+pipeline_folder)
pipeline_results = dict()

for pipeline_name in pipeline_list:

  # load the model
  df_transformed, topics, vocabulary = run_pipeline(pipeline_name, pipeline_folder, df_docs)
  
  # get the cluster for each customer and clusters size
  df_transformed, cluster_sumary = clust_info(df_transformed)
  
  # index the words for undestanding
  topics = word_indexer(topics, vocabulary)
  
  # convert arrays to zip term/term_weight, explode to rows, cols
  df_clusters = transform_clus_data(topics, df_lang, cluster_sumary)
  
  # cluster by customer
  cust_clust = (df_transformed
                .select('card_id','cluster')
                .join(df_cust, on='card_id', how='inner')
               )
  
  # get the cluster items per category
  clust_cat = items_clust(cust_clust, df_transactions, df_old_cat, df_words)

  
  pipeline_results[pipeline_name] = [df_clusters.toPandas(), cluster_sumary.toPandas(), 
                                     cust_clust.toPandas(), clust_cat.toPandas()]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## REPORT

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### HTML

# COMMAND ----------

def graph_treemap(name, results, language):
  
  fig = px.treemap(results, path=[px.Constant(name), 'cluster_info', f'word_{language}'], values='termWeights')
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

  df = clust_size.sort_values('cluster')

  hovertemplate= """
  クラスター: %{label}<br>
  客様の数: %{value}<br>
  比率: %{percent}
  <extra></extra>
  """

  texttemplate= """
  クラスター: %{label}<br>
  %{value}<br>
  %{percent}
  """

  fig = go.Figure(go.Pie(labels=df['cluster'], legendgrouptitle=dict(text='クラスター サイズ順'), 
                         textfont=dict(size=10),
                         textposition='inside',
                         texttemplate=texttemplate,
                         hovertemplate=hovertemplate,
                         values=df['size'], 
                         sort=True,
                        ))

  fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
  
  html = fig.to_html(include_plotlyjs=False,
                            include_mathjax=False, 
                            full_html=False, 
                            default_width='90%', 
                            default_height='80%',
                            
                           )
  html = html.replace('class="plotly-graph-div"', 'class="plotly-graph-div pie_chart"')
  
  return html

# COMMAND ----------

def graph_gender_all(cust_clust):
  
  data = cust_clust

  men = (data[data['gender'] == '男']
         .groupby('age')
         .agg({'card_id':'count'})
         .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
         .reset_index()
        )
  women = (data[data['gender'] == '女']
           .groupby('age')
           .agg({'card_id':'count'})
           .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
           .reset_index()
          )
  
  fig = make_subplots(shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=1,
                    y_title='人数',
                    subplot_titles=['女性','男性'],
                    rows=1, cols=2,
                      
                   )

  bar_men = go.Bar(x=men['age'], y=men['card_id'])
  bar_women = go.Bar(x=women['age'], y=women['card_id'])

  fig.add_trace(bar_men, row=1, col=2)
  fig.add_trace(bar_women, row=1, col=1)

  fig.update_layout(showlegend=False, title_text="性別分布", height=600)
  
  html = fig.to_html(include_plotlyjs=False,
                            include_mathjax=False, 
                            full_html=False, 
                            default_width='90%', 
                            default_height='80%',
                            
                           )
  return html

# COMMAND ----------

def graph_gender(all_cust, clust_cust):
  
  allmen = (all_cust[all_cust['gender'] == '男']
         .groupby('age')
         .agg({'card_id':'count'})
         .rename(columns={'card_id':'total_cust'})
         .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
        )
  
  allwomen = (all_cust[all_cust['gender'] == '女']
           .groupby('age')
           .agg({'card_id':'count'})
           .rename(columns={'card_id':'total_cust'})
           .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
          )
  
  men = (clust_cust[clust_cust['gender'] == '男']
         .groupby('age')
         .agg({'card_id':'count'})
         .rename(columns={'card_id':'clust_cust'})
         .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
        )
  women = (clust_cust[clust_cust['gender'] == '女']
           .groupby('age')
           .agg({'card_id':'count'})
           .rename(columns={'card_id':'clust_cust'})
           .reindex(['20代未満', '20代' , '30代', '40代', '50代', 
                   '60代', '70代', '80代', '90代以上', '世代不明'])
          )
  
  
  men_ratio = (men
         .join(allmen)
         .assign(percent = lambda x: x['clust_cust']/x['total_cust']*100)
         .reset_index()
        )
  
  women_ratio = (women
           .join(allwomen)
           .assign(percent = lambda x: x['clust_cust']/x['total_cust']*100)
           .reset_index()
        )
  row_titles = ['人数', '人比']
  fig = make_subplots(horizontal_spacing=0.01, 
                      vertical_spacing=0.05,  
                      shared_yaxes=True,
                      shared_xaxes=True,
                      row_titles=row_titles,
                      column_titles=['女性','男性'],
                      rows=2, cols=2,
                      
                   )
  
  men = men.reset_index()
  women = women.reset_index()

  bar_men = go.Bar(x=men['age'], y=men['clust_cust'])
  bar_women = go.Bar(x=women['age'], y=women['clust_cust'])
  bar_men_ratio = go.Bar(x=men_ratio['age'], y=men_ratio['percent'])
  bar_women_ratio = go.Bar(x=women_ratio['age'], y=women_ratio['percent'])

  fig.add_trace(bar_men, row=1, col=2)
  fig.add_trace(bar_women, row=1, col=1)
  fig.add_trace(bar_men_ratio, row=2, col=2)
  fig.add_trace(bar_women_ratio, row=2, col=1)

  fig.update_layout(showlegend=False, title_text="性別分布", height=600)
  
  fig.for_each_annotation(lambda x:  x.update(x = -0.07) if x.text in row_titles else())

  
  html = fig.to_html(include_plotlyjs=False,
                            include_mathjax=False, 
                            full_html=False, 
                            default_width='90%', 
                            default_height='80%',
                            
                           )
  return html

# COMMAND ----------

def html_head(html_file, title):
  
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
  
  html_file.write('<div class="space"></div>''\n')
  html_file.write(f'<h1>{title}</h1>''\n')
  
  return html_file

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/reports/11_30v1

# COMMAND ----------

# to export the cluster of each customer
df_excel = df_cust.select('card_id')
language = 'both'

for pipeline_no, (pipeline_name, results)  in enumerate(pipeline_results.items()):
  pipeline_name = pipeline_name.replace(' ', '-')
  
  file_name = f'PV1{pipeline_name}_{date}'
  loc = f'../../dbfs/FileStore/files/LDA_MODELS/reports/{date}v1/'
  title = 'フレスコークラスタリング'
  
  ### EXTRACT DATA ##################################################
  df_topics = results[0]
  df_clust_summary = results[1]
  df_customers = results[2]
  df_cat_clust = results[3]
  
  ### EXCEL FOR CATEGORIES ##########################################
  df_cat_clust.to_csv(f"{loc}{file_name}_purchase_data.csv")
  
  ### CLUSTER SUMMARY ###############################################
  df_clust_summary.to_csv(f"{loc}{file_name}_clust_summary.csv")
 
  #### HTML #########################################################
  html_file = open(f"{loc}{file_name}_overview.html",'w')
  html_file = html_head(html_file, title)
  
  # piechart
  html_pie = graph_pie(df_clust_summary)
  # main treemap
  html_treemap = graph_treemap(' ', df_topics, language)
  # gender chart
  html_gender = graph_gender_all(df_customers)
  
  html_file.write(f'<p>{pipeline_name}</p>''\n')
  html_file.write(html_pie+'\n')
  html_file.write(html_treemap+'\n')
  html_file.write(html_gender+'\n')
  
  # cluster specific overview 
  nclusters = df_clust_summary['cluster'].unique()
  nclusters.sort()
  
  for cluster in nclusters:
    # cluster treemap
    
    clust_data = df_topics.query(f'cluster == {cluster}')
    clust_customers = df_customers.query(f'cluster == {cluster}')

    html_file.write(f'<h3>Cluster: {cluster}</h3>''\n')
    
    name = f'Cluster: {cluster}'
    html_clustreemap = graph_treemap(name, clust_data, language)
    html_clusgender_ratio = graph_gender(df_customers, clust_customers)

    # cluster gender distribution
    html_file.write(html_clustreemap+'\n')
    html_file.write(html_clusgender_ratio+'\n')
    html_file.write('<div class="space"></div>''\n')

  #### CUSTOMERS CLUSTER #####################################################
  df_customers = (spark
                  .createDataFrame(df_customers)
                  .select('card_id','cluster')
                  .withColumnRenamed('cluster', pipeline_name))
  
  df_excel = df_excel.join(df_customers, on='card_id', how='inner')
  
  html_file.write("\n</div></body></html>")
  html_file.close()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### EXCEL

# COMMAND ----------

df_excel.coalesce(1).write.csv(f'/FileStore/files/LDA_MODELS/reports/11_30v1/V1_CLUST_CUST_{pipeline_list[0]}.csv', header=True)

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/reports/11_30v1/V1_CLUST_CUST_NO_CLUST16.csv/

# COMMAND ----------

ls ../../dbfs/FileStore/files/LDA_MODELS/reports/11_30v1/

# COMMAND ----------


