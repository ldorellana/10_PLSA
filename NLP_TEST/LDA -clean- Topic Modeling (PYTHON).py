# Databricks notebook source
# MAGIC %md #Topic Modeling with Latent Dirichlet Allocation
# MAGIC 
# MAGIC This notebook will provide a brief algorithm summary, links for further reading, and an example of how to use LDA for Topic Modeling.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Reviewing our Data
# MAGIC 
# MAGIC In this example, we will use the mini [20 Newsgroups dataset](http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html), which is a random subset of the original 20 Newsgroups dataset. Each newsgroup is stored in a subdirectory, with each article stored as a separate file.
# MAGIC 
# MAGIC The mini dataset consists of 100 articles from the following 20 Usenet newsgroups:
# MAGIC 
# MAGIC     alt.atheism
# MAGIC     comp.graphics
# MAGIC     comp.os.ms-windows.misc
# MAGIC     comp.sys.ibm.pc.hardware
# MAGIC     comp.sys.mac.hardware
# MAGIC     comp.windows.x
# MAGIC     misc.forsale
# MAGIC     rec.autos
# MAGIC     rec.motorcycles
# MAGIC     rec.sport.baseball
# MAGIC     rec.sport.hockey
# MAGIC     sci.crypt
# MAGIC     sci.electronics
# MAGIC     sci.med
# MAGIC     sci.space
# MAGIC     soc.religion.christian
# MAGIC     talk.politics.guns
# MAGIC     talk.politics.mideast
# MAGIC     talk.politics.misc
# MAGIC     talk.religion.misc
# MAGIC 
# MAGIC Some of the newsgroups seem pretty similar on first glance, such as *comp.sys.ibm.pc.hardware* and *comp.sys.mac.hardware*, which may affect our results.

# COMMAND ----------

# MAGIC %md ##Loading the Data and Data Cleaning
# MAGIC 
# MAGIC We will use the wget command to download the file, and read in the data using wholeTextFiles().

# COMMAND ----------

# MAGIC %md
# MAGIC The `wholeTextFiles()` command will read in the entire directory of text files, and return a key-value pair of (filePath, fileContent).
# MAGIC 
# MAGIC As we do not need the file paths in this example, we will apply a map function to extract the file contents, and then convert everything to lowercase.

# COMMAND ----------

import pyspark.sql.functions as f


corpus = (sc.wholeTextFiles('/tmp/mini_newsgroups/*')
          .toDF()
          .drop('_1')
          .withColumn('_2', f.lower('_2'))
          .withColumnRenamed('_2','docs')
         )

# display(corpus.head(2))

# COMMAND ----------

# MAGIC %md Note that the document begins with a header containing some metadata that we don't need, and we are only interested in the body of the document. We can do a bit of simple data cleaning here by removing the metadata of each document, which reduces the noise in our dataset. This is an important step as the accuracy of our models depend greatly on the quality of data used.

# COMMAND ----------

# drop the metadata
drop_headerUDF = f.udf(lambda z: ' '.join(z[1:]))

corpus_clean = (corpus
          .withColumn('docs', f.split('docs', "\\n\\n")) # separate the metadata
          .withColumn('docs', drop_headerUDF(f.col('docs'))) # drop the metadata
          .withColumn('id', f.monotonically_increasing_id()) # add id column for each doc
         )

# display(corpus_clean.head(2))

# COMMAND ----------

# MAGIC %md To use the convenient [Feature extraction and transformation APIs](http://spark.apache.org/docs/latest/ml-features.html), we will convert our RDD into a DataFrame.
# MAGIC 
# MAGIC We will also create an ID for every document.

# COMMAND ----------

# MAGIC %md ## Text Tokenization
# MAGIC 
# MAGIC We will use the `RegexTokenizer` to split each document into tokens. We can `setMinTokenLength()` here to indicate a minimum token length, and filter away all tokens that fall below the minimum.

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

tokenizer = RegexTokenizer(pattern="[\\W_]+", 
                           minTokenLength=4, 
                           inputCol='docs', 
                           outputCol='tokens')

tokenized_df = tokenizer.transform(corpus_clean)

# display(tokenized_df.select(['tokens','docs']).head(2))

# COMMAND ----------

# MAGIC %md ## Remove Stopwords
# MAGIC 
# MAGIC We can easily remove stopwords using the `StopWordsRemover()`. If a list of stopwords is not provided, the `StopWordsRemover()` will use [this list of stopwords](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words) by default. You can use `getStopWords()` to see the list of stopwords that will be used.
# MAGIC 
# MAGIC In this example, we will specify a list of stopwords for the `StopWordsRemover()` to use. We do this so that we can add on to the list later on.

# COMMAND ----------

stopwords = sc.textFile("/tmp/stopwords").collect()

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(stopWords=stopwords, 
                           inputCol='tokens', 
                           outputCol='fil_tokens')

fil_df = remover.transform(tokenized_df)

# display(fil_df.select(['fil_tokens','tokens']).head(2))

# COMMAND ----------

# MAGIC %md ## Vector of Token Counts
# MAGIC 
# MAGIC LDA takes in a vector of token counts as input. We can use the `CountVectorizer()` to easily convert our text documents into vectors of token counts.
# MAGIC 
# MAGIC The `CountVectorizer` will return (VocabSize, Array(Indexed Tokens), Array(Token Frequency))
# MAGIC 
# MAGIC Two handy parameters to note:
# MAGIC   - `setMinDF`: Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
# MAGIC   - `setMinTF`: Specifies the minimumn number of times a term has to appear in a document to be included in the vocabulary.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF

#DF = (max/min) number or fraction of documents the word appreas
#TF = (min) number or fraction of times the word appears in a document

countvectorizer = (CountVectorizer(inputCol='fil_tokens', 
                                  outputCol='features', 
                                  vocabSize=10_000, 
                                  minDF=5)
                   .fit(fil_df)
                  )


counted_df = countvectorizer.transform(fil_df)

# COMMAND ----------

display(counted_df)

# COMMAND ----------

# make use of IDF to 

weigthed_df = (IDF(inputCol='features', outputCol='w_features')
               .fit(counted_df)
               .transform(counted_df)
              )

# display(weigthed_df.select(['w_features','features']).head(2))

# COMMAND ----------

# MAGIC %md ## Create LDA model with Online Variational Bayes
# MAGIC 
# MAGIC We will now set the parameters for LDA. We will use the `OnlineLDAOptimizer()` here, which implements Online Variational Bayes.
# MAGIC 
# MAGIC Choosing the number of topics for your LDA model requires a bit of domain knowledge. As we know that there are 20 unique newsgroups in our dataset, we will set numTopics to be 20.

# COMMAND ----------

# MAGIC %md
# MAGIC We will set the parameters needed to build our LDA model. We can also `setMiniBatchFraction` for the `OnlineLDAOptimizer`, which sets the fraction of corpus sampled and used at each iteration. In this example, we will set this to 0.8.

# COMMAND ----------

from  pyspark.ml.clustering import LDA

numTopics = 10

lda_model = (LDA(featuresCol='w_features', 
                 k=numTopics,
                 maxIter=20, 
                 seed=20, 
                 optimizer='em')
             .fit(weigthed_df))

# COMMAND ----------

# MAGIC %md ## Review Topics
# MAGIC 
# MAGIC We can now review the results of our LDA model. We will print out all 20 topics with their corresponding term probabilities.
# MAGIC 
# MAGIC Note that you will get slightly different results every time you run an LDA model since LDA includes some randomization.

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType

# Review Results of LDA model with Online Variational Bayes
topicIndices = lda_model.describeTopics(maxTermsPerTopic = 200)
vocabList = countvectorizer.vocabulary

def term_word(termIndices):
  
  words = [vocabList[term] for term in termIndices]
  return words
  

term_wordUDF = f.udf(term_word, returnType=ArrayType(StringType()))

topics = (topicIndices.withColumn('terms', term_wordUDF(f.col('termIndices'))))

# for topic in topics.toLocalIterator():
#   print(f'TOPIC {topic.topic}')
    
#   for term, weigth in zip(topic.termIndices, topic.termWeights):
#     print(f'{vocabList[term]} \t {weigth}')
#   print('===============')

# COMMAND ----------

# MAGIC %md Going through the results, you may notice that some of the topic words returned are actually stopwords that are specific to our dataset (for eg: "writes", "article"...). Let's try improving our model.

# COMMAND ----------

transformed_df = lda_model.transform(weigthed_df)
# display(transformed_df.head(2))

# COMMAND ----------

# MAGIC %md We managed to get better results here. Your results will likely differ because of how LDA works and the randomness inherent in the model however we can see some strength in the general topics.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CONVERTING RESUTLS

# COMMAND ----------

pandas_df = (topics
        .withColumn('zipped', f.arrays_zip('terms','termWeights'))
        .withColumn('exp', f.explode('zipped'))
        .select('topic','exp.terms', 'exp.termWeights')
        .toPandas()
       )

# COMMAND ----------

import pandas as pd
import plotly.express as px
df = pandas_df
fig = px.treemap(df, path=[px.Constant("TOPICS"), 'topic', 'terms'], values='termWeights')
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# COMMAND ----------


