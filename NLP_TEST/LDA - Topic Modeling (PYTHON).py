# Databricks notebook source
# MAGIC %md #Topic Modeling with Latent Dirichlet Allocation
# MAGIC 
# MAGIC This notebook will provide a brief algorithm summary, links for further reading, and an example of how to use LDA for Topic Modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Algorithm Summary
# MAGIC - **Task**: Identify topics from a collection of text documents
# MAGIC - **Input**: Vectors of word counts
# MAGIC - **Optimizers**: 
# MAGIC     - EMLDAOptimizer using [Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
# MAGIC     - OnlineLDAOptimizer using Iterative Mini-Batch Sampling for [Online Variational Bayes](https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Links
# MAGIC - Spark API docs
# MAGIC   - RDD-based `spark.mllib` API (used in this notebook)
# MAGIC     - Scala: [LDA](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.LDA)
# MAGIC     - Python: [LDA](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.LDA)
# MAGIC     - [MLlib Programming Guide](http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda)
# MAGIC   - DataFrame-based `spark.ml` API (same functionality, but using DataFrames for input)
# MAGIC     - Scala: [LDA](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.LDA)
# MAGIC     - Python: [LDA](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.LDA)
# MAGIC     - [ML Programming Guide](http://spark.apache.org/docs/latest/ml-clustering.html#latent-dirichlet-allocation-lda)
# MAGIC - [ML Feature Extractors & Transformers](http://spark.apache.org/docs/latest/ml-features.html)
# MAGIC - [Wikipedia: Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

# COMMAND ----------

# MAGIC %md ## Topic Modeling Example
# MAGIC 
# MAGIC This is an outline of our Topic Modeling workflow. Feel free to jump to any subtopic to find out more.
# MAGIC - Dataset Review
# MAGIC - Loading the Data and Data Cleaning
# MAGIC - Text Tokenization
# MAGIC - Remove Stopwords
# MAGIC - Vector of Token Counts
# MAGIC - Create LDA model with Online Variational Bayes
# MAGIC - Review Topics
# MAGIC - Model Tuning - Refilter Stopwords
# MAGIC - Create LDA model with Expectation Maximization
# MAGIC - Visualize Results

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

# MAGIC %sh wget http://kdd.ics.uci.edu/databases/20newsgroups/mini_newsgroups.tar.gz -O /tmp/newsgroups.tar.gz

# COMMAND ----------

# MAGIC %md Untar the file into the /tmp/ folder.

# COMMAND ----------

# MAGIC %sh tar xvfz /tmp/newsgroups.tar.gz -C /tmp/

# COMMAND ----------

# MAGIC %md The below cell takes about 10mins to run.

# COMMAND ----------

# MAGIC %fs cp -r file:/tmp/mini_newsgroups dbfs:/tmp/mini_newsgroups

# COMMAND ----------

# MAGIC %md
# MAGIC The `wholeTextFiles()` command will read in the entire directory of text files, and return a key-value pair of (filePath, fileContent).
# MAGIC 
# MAGIC As we do not need the file paths in this example, we will apply a map function to extract the file contents, and then convert everything to lowercase.

# COMMAND ----------

ls /tmp/mini_newsgroups/comp.graphics/38473  

# COMMAND ----------

import pyspark.sql.functions as f
corpus = (sc.wholeTextFiles('/tmp/mini_newsgroups/*')
          .toDF()
          .drop('_1')
          .withColumn('_2', f.lower('_2'))
         )

# COMMAND ----------

display(corpus)

# COMMAND ----------

# MAGIC %md Note that the document begins with a header containing some metadata that we don't need, and we are only interested in the body of the document. We can do a bit of simple data cleaning here by removing the metadata of each document, which reduces the noise in our dataset. This is an important step as the accuracy of our models depend greatly on the quality of data used.

# COMMAND ----------

def drop_header(txtlist):
  return ' '.join(txtlist[1:])

drop_headerUDF = f.udf(lambda z: drop_header(z))

corpus = (corpus
          .withColumn('_2', f.split('_2', "\\n\\n"))
          .withColumn('_2', drop_headerUDF(f.col('_2')))
         )

# COMMAND ----------

# MAGIC %md To use the convenient [Feature extraction and transformation APIs](http://spark.apache.org/docs/latest/ml-features.html), we will convert our RDD into a DataFrame.
# MAGIC 
# MAGIC We will also create an ID for every document.

# COMMAND ----------

corpus_df = (corpus
             .withColumn('id', f.monotonically_increasing_id())
             .withColumnRenamed('_2', 'corpus')
            )

# COMMAND ----------

display(corpus_df)

# COMMAND ----------

# MAGIC %md ## Text Tokenization
# MAGIC 
# MAGIC We will use the `RegexTokenizer` to split each document into tokens. We can `setMinTokenLength()` here to indicate a minimum token length, and filter away all tokens that fall below the minimum.

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

tokenizer = (RegexTokenizer()
  .setPattern("[\\W_]+")
  .setMinTokenLength(4) # Filter away tokens with length < 4
  .setInputCol("corpus")
  .setOutputCol("tokens")
            )
  
tokenized_df = tokenizer.transform(corpus_df)

# COMMAND ----------

display(tokenized_df)

# COMMAND ----------

# MAGIC %md ## Remove Stopwords
# MAGIC 
# MAGIC We can easily remove stopwords using the `StopWordsRemover()`. If a list of stopwords is not provided, the `StopWordsRemover()` will use [this list of stopwords](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words) by default. You can use `getStopWords()` to see the list of stopwords that will be used.
# MAGIC 
# MAGIC In this example, we will specify a list of stopwords for the `StopWordsRemover()` to use. We do this so that we can add on to the list later on.

# COMMAND ----------

# MAGIC %sh wget http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words -O /tmp/stopwords

# COMMAND ----------

# MAGIC %fs cp file:/tmp/stopwords dbfs:/tmp/stopwords

# COMMAND ----------

stopwords = sc.textFile("/tmp/stopwords").collect()

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

# Set params for StopWordsRemover
remover = (StopWordsRemover()
  .setStopWords(stopwords) # This parameter is optional
  .setInputCol("tokens")
  .setOutputCol("filtered"))

# Create new DF with Stopwords removed
filtered_df = remover.transform(tokenized_df)

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

# Set params for CountVectorizer
vectorizer = (CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  .setVocabSize(10000)
  .setMinDF(5)
  .fit(filtered_df))

# COMMAND ----------

count_df = vectorizer.transform(filtered_df)

weigthed_df = (IDF()
            .setInputCol("features")
            .setOutputCol("features_w")
            .fit(count_df)
            .transform(count_df)
           )

# COMMAND ----------

display(weigthed_df)

# COMMAND ----------

# MAGIC %md ## Create LDA model with Online Variational Bayes
# MAGIC 
# MAGIC We will now set the parameters for LDA. We will use the `OnlineLDAOptimizer()` here, which implements Online Variational Bayes.
# MAGIC 
# MAGIC Choosing the number of topics for your LDA model requires a bit of domain knowledge. As we know that there are 20 unique newsgroups in our dataset, we will set numTopics to be 20.

# COMMAND ----------

numTopics = 10

# COMMAND ----------

# MAGIC %md
# MAGIC We will set the parameters needed to build our LDA model. We can also `setMiniBatchFraction` for the `OnlineLDAOptimizer`, which sets the fraction of corpus sampled and used at each iteration. In this example, we will set this to 0.8.

# COMMAND ----------

# from pyspark.ml.linalg import Vectors, SparseVector

# list_to_vetorUDF = f.udf(lambda l: [l['lenght'],l['indices'], l['values']])

# display(count_df.withColumn('features_w', list_to_vetorUDF(f.col('features_w'))))

# #lda_vector = count_df[['id','features']].rdd.map()

# COMMAND ----------

from  pyspark.ml.clustering import LDA

# Set LDA params

lda_model = LDA(featuresCol='features_w', maxIter=10, seed=32, k=10, optimizer='em').fit(weigthed_df)

# COMMAND ----------

# MAGIC %md Create the LDA model with Online Variational Bayes.

# COMMAND ----------

# MAGIC %md ## Review Topics
# MAGIC 
# MAGIC We can now review the results of our LDA model. We will print out all 20 topics with their corresponding term probabilities.
# MAGIC 
# MAGIC Note that you will get slightly different results every time you run an LDA model since LDA includes some randomization.

# COMMAND ----------

# Review Results of LDA model with Online Variational Bayes
topicIndices = lda_model.describeTopics(maxTermsPerTopic = 10)
vocabList = vectorizer.vocabulary

def term_word(termIndices):
  words = []
  for term in termIndices:
    words.append(vocabList[term])
  return words

term_wordUDF = f.udf(term_word)
topics = (topicIndices.withColumn('terms', term_wordUDF(f.col('termIndices'))))

# COMMAND ----------

for topic in topics.toLocalIterator():
  print(f'TOPIC {topic.topic}')
    
  for term, weigth in zip(topic.termIndices, topic.termWeights):
    print(f'{vocabList[term]} \t {weigth}')
  print('===============')

# COMMAND ----------

# MAGIC %md Going through the results, you may notice that some of the topic words returned are actually stopwords that are specific to our dataset (for eg: "writes", "article"...). Let's try improving our model.

# COMMAND ----------

# MAGIC %md ## Model Tuning - Refilter Stopwords
# MAGIC 
# MAGIC We will try to improve the results of our model by identifying some stopwords that are specific to our dataset. We will filter these stopwords out and rerun our LDA model to see if we get better results.

# COMMAND ----------

add_stopwords = ["article", "writes", "entry", "date", "udel", "said", "tell", "think", "know", "just", "newsgroup", "line", "like", "does", "going", "make", "thanks"]

# COMMAND ----------

new_stopwords = stopwords.append(add_stopwords)

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.feature.StopWordsRemover
# MAGIC 
# MAGIC // Set Params for StopWordsRemover with new_stopwords
# MAGIC val remover = new StopWordsRemover()
# MAGIC   .setStopWords(new_stopwords)
# MAGIC   .setInputCol("tokens")
# MAGIC   .setOutputCol("filtered")
# MAGIC 
# MAGIC // Create new df with new list of stopwords removed
# MAGIC val new_filtered_df = remover.transform(tokenized_df)

# COMMAND ----------

# MAGIC %scala
# MAGIC // Set Params for CountVectorizer
# MAGIC val vectorizer = new CountVectorizer()
# MAGIC   .setInputCol("filtered")
# MAGIC   .setOutputCol("features")
# MAGIC   .setVocabSize(10000)
# MAGIC   .setMinDF(5)
# MAGIC   .fit(new_filtered_df)
# MAGIC 
# MAGIC // Create new df of countVectors
# MAGIC val new_countVectors = vectorizer.transform(new_filtered_df).select("id", "features")

# COMMAND ----------

# MAGIC %scala
# MAGIC // Convert DF to RDD
# MAGIC val new_lda_countVector = new_countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }

# COMMAND ----------

# MAGIC %md We will also increase `MaxIterations` to 10 to see if we get better results.

# COMMAND ----------

# MAGIC %scala
# MAGIC // Set LDA parameters
# MAGIC val new_lda = new LDA()
# MAGIC   .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
# MAGIC   .setK(numTopics)
# MAGIC   .setMaxIterations(10)
# MAGIC   .setDocConcentration(-1) // use default values
# MAGIC   .setTopicConcentration(-1) // use default values

# COMMAND ----------

# MAGIC %scala
# MAGIC // Create LDA model with stopwords refiltered
# MAGIC val new_ldaModel = new_lda.run(new_lda_countVector)
# MAGIC // this will take a while to run!

# COMMAND ----------

# MAGIC %scala
# MAGIC val topicIndices = new_ldaModel.describeTopics(maxTermsPerTopic = 5)
# MAGIC val vocabList = vectorizer.vocabulary
# MAGIC val topics = topicIndices.map { case (terms, termWeights) =>
# MAGIC   terms.map(vocabList(_)).zip(termWeights)
# MAGIC }
# MAGIC println(s"$numTopics topics:")
# MAGIC topics.zipWithIndex.foreach { case (topic, i) =>
# MAGIC   println(s"TOPIC $i")
# MAGIC   topic.foreach { case (term, weight) => println(s"$term\t$weight") }
# MAGIC   println(s"==========")
# MAGIC }

# COMMAND ----------

# MAGIC %md We managed to get better results here. Your results will likely differ because of how LDA works and the randomness inherent in the model however we can see some strength in the general topics.

# COMMAND ----------

# MAGIC %md ## Create LDA model with Expectation Maximization
# MAGIC 
# MAGIC Let's try creating an LDA model with Expectation Maximization on the data that has been refiltered for additional stopwords. We will also increase `MaxIterations` here to 100 to see if that improves results.

# COMMAND ----------

# MAGIC %scala
# MAGIC // Set LDA parameters
# MAGIC val em_lda = new LDA()
# MAGIC   .setOptimizer("em")
# MAGIC   .setK(numTopics)
# MAGIC   .setMaxIterations(100)
# MAGIC   .setDocConcentration(-1) // use default values
# MAGIC   .setTopicConcentration(-1) // use default values

# COMMAND ----------

# MAGIC %scala
# MAGIC val em_ldaModel = em_lda.run(new_lda_countVector)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the `EMLDAOptimizer` produces a `DistributedLDAModel`, which stores not only the inferred topics but also the full training corpus and topic distributions for each document in the training corpus.

# COMMAND ----------

# MAGIC %scala
# MAGIC val topicIndices = em_ldaModel.describeTopics(maxTermsPerTopic = 5)
# MAGIC val vocabList = vectorizer.vocabulary
# MAGIC val topics = topicIndices.map { case (terms, termWeights) =>
# MAGIC   terms.map(vocabList(_)).zip(termWeights)
# MAGIC }
# MAGIC println(s"$numTopics topics:")
# MAGIC topics.zipWithIndex.foreach { case (topic, i) =>
# MAGIC   println(s"TOPIC $i")
# MAGIC   topic.foreach { case (term, weight) => println(s"$term\t$weight") }
# MAGIC   println(s"==========")
# MAGIC }

# COMMAND ----------

# MAGIC %md We managed to get better results here. Your results will likely differ because of how LDA works and the randomness inherent in the model however we can see some improvements in the topics.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To improve our results further, we could employ some of the below methods:
# MAGIC - Refilter data for additional data-specific stopwords
# MAGIC - Use Stemming or Lemmatization to preprocess data
# MAGIC - Experiment with a smaller number of topics, since some of these topics in the 20 Newsgroups are pretty similar
# MAGIC - Increase model's MaxIterations

# COMMAND ----------

# MAGIC %md ## Visualize Results
# MAGIC 
# MAGIC We will try visualizing the results obtained from the EM LDA model with a d3 bubble chart.

# COMMAND ----------

# MAGIC %scala
# MAGIC // Zip topic terms with topic IDs
# MAGIC val termArray = topics.zipWithIndex

# COMMAND ----------

# MAGIC %scala
# MAGIC // Transform data into the form (term, probability, topicId)
# MAGIC val termRDD = sc.parallelize(termArray)
# MAGIC val termRDD2 =termRDD.flatMap( (x: (Array[(String, Double)], Int)) => {
# MAGIC   val arrayOfTuple = x._1
# MAGIC   val topicId = x._2
# MAGIC   arrayOfTuple.map(el => (el._1, el._2, topicId))
# MAGIC })

# COMMAND ----------

# MAGIC %scala
# MAGIC // Create DF with proper column names
# MAGIC val termDF = termRDD2.toDF.withColumnRenamed("_1", "term").withColumnRenamed("_2", "probability").withColumnRenamed("_3", "topicId")

# COMMAND ----------

# MAGIC %scala
# MAGIC display(termDF)

# COMMAND ----------

# MAGIC %md We will convert the DataFrame into a JSON format, which will be passed into d3.

# COMMAND ----------

# MAGIC %scala
# MAGIC // Create JSON data
# MAGIC val rawJson = termDF.toJSON.collect().mkString(",\n")

# COMMAND ----------

# MAGIC %md We are now ready to use D3 on the rawJson data.

# COMMAND ----------

# MAGIC %scala
# MAGIC displayHTML(s"""
# MAGIC <!DOCTYPE html>
# MAGIC <meta charset="utf-8">
# MAGIC <style>
# MAGIC 
# MAGIC circle {
# MAGIC   fill: rgb(31, 119, 180);
# MAGIC   fill-opacity: 0.5;
# MAGIC   stroke: rgb(31, 119, 180);
# MAGIC   stroke-width: 1px;
# MAGIC }
# MAGIC 
# MAGIC .leaf circle {
# MAGIC   fill: #ff7f0e;
# MAGIC   fill-opacity: 1;
# MAGIC }
# MAGIC 
# MAGIC text {
# MAGIC   font: 14px sans-serif;
# MAGIC }
# MAGIC 
# MAGIC </style>
# MAGIC <body>
# MAGIC <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
# MAGIC <script>
# MAGIC 
# MAGIC var json = {
# MAGIC  "name": "data",
# MAGIC  "children": [
# MAGIC   {
# MAGIC      "name": "topics",
# MAGIC      "children": [
# MAGIC       ${rawJson}
# MAGIC      ]
# MAGIC     }
# MAGIC    ]
# MAGIC };
# MAGIC 
# MAGIC var r = 1500,
# MAGIC     format = d3.format(",d"),
# MAGIC     fill = d3.scale.category20c();
# MAGIC 
# MAGIC var bubble = d3.layout.pack()
# MAGIC     .sort(null)
# MAGIC     .size([r, r])
# MAGIC     .padding(1.5);
# MAGIC 
# MAGIC var vis = d3.select("body").append("svg")
# MAGIC     .attr("width", r)
# MAGIC     .attr("height", r)
# MAGIC     .attr("class", "bubble");
# MAGIC 
# MAGIC   
# MAGIC var node = vis.selectAll("g.node")
# MAGIC     .data(bubble.nodes(classes(json))
# MAGIC     .filter(function(d) { return !d.children; }))
# MAGIC     .enter().append("g")
# MAGIC     .attr("class", "node")
# MAGIC     .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
# MAGIC     color = d3.scale.category20();
# MAGIC   
# MAGIC   node.append("title")
# MAGIC       .text(function(d) { return d.className + ": " + format(d.value); });
# MAGIC 
# MAGIC   node.append("circle")
# MAGIC       .attr("r", function(d) { return d.r; })
# MAGIC       .style("fill", function(d) {return color(d.topicName);});
# MAGIC 
# MAGIC var text = node.append("text")
# MAGIC     .attr("text-anchor", "middle")
# MAGIC     .attr("dy", ".3em")
# MAGIC     .text(function(d) { return d.className.substring(0, d.r / 3)});
# MAGIC   
# MAGIC   text.append("tspan")
# MAGIC       .attr("dy", "1.2em")
# MAGIC       .attr("x", 0)
# MAGIC       .text(function(d) {return Math.ceil(d.value * 10000) /10000; });
# MAGIC 
# MAGIC // Returns a flattened hierarchy containing all leaf nodes under the root.
# MAGIC function classes(root) {
# MAGIC   var classes = [];
# MAGIC 
# MAGIC   function recurse(term, node) {
# MAGIC     if (node.children) node.children.forEach(function(child) { recurse(node.term, child); });
# MAGIC     else classes.push({topicName: node.topicId, className: node.term, value: node.probability});
# MAGIC   }
# MAGIC 
# MAGIC   recurse(null, root);
# MAGIC   return {children: classes};
# MAGIC }
# MAGIC </script>
# MAGIC """)
