// Databricks notebook source
// MAGIC %md #Topic Modeling with Latent Dirichlet Allocation
// MAGIC 
// MAGIC This notebook will provide a brief algorithm summary, links for further reading, and an example of how to use LDA for Topic Modeling.

// COMMAND ----------

// MAGIC %md
// MAGIC ##Algorithm Summary
// MAGIC - **Task**: Identify topics from a collection of text documents
// MAGIC - **Input**: Vectors of word counts
// MAGIC - **Optimizers**: 
// MAGIC     - EMLDAOptimizer using [Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
// MAGIC     - OnlineLDAOptimizer using Iterative Mini-Batch Sampling for [Online Variational Bayes](https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Links
// MAGIC - Spark API docs
// MAGIC   - RDD-based `spark.mllib` API (used in this notebook)
// MAGIC     - Scala: [LDA](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.LDA)
// MAGIC     - Python: [LDA](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.LDA)
// MAGIC     - [MLlib Programming Guide](http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda)
// MAGIC   - DataFrame-based `spark.ml` API (same functionality, but using DataFrames for input)
// MAGIC     - Scala: [LDA](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.clustering.LDA)
// MAGIC     - Python: [LDA](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.LDA)
// MAGIC     - [ML Programming Guide](http://spark.apache.org/docs/latest/ml-clustering.html#latent-dirichlet-allocation-lda)
// MAGIC - [ML Feature Extractors & Transformers](http://spark.apache.org/docs/latest/ml-features.html)
// MAGIC - [Wikipedia: Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

// COMMAND ----------

// MAGIC %md ## Topic Modeling Example
// MAGIC 
// MAGIC This is an outline of our Topic Modeling workflow. Feel free to jump to any subtopic to find out more.
// MAGIC - Dataset Review
// MAGIC - Loading the Data and Data Cleaning
// MAGIC - Text Tokenization
// MAGIC - Remove Stopwords
// MAGIC - Vector of Token Counts
// MAGIC - Create LDA model with Online Variational Bayes
// MAGIC - Review Topics
// MAGIC - Model Tuning - Refilter Stopwords
// MAGIC - Create LDA model with Expectation Maximization
// MAGIC - Visualize Results

// COMMAND ----------

// MAGIC %md 
// MAGIC 
// MAGIC ## Reviewing our Data
// MAGIC 
// MAGIC In this example, we will use the mini [20 Newsgroups dataset](http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html), which is a random subset of the original 20 Newsgroups dataset. Each newsgroup is stored in a subdirectory, with each article stored as a separate file.
// MAGIC 
// MAGIC The mini dataset consists of 100 articles from the following 20 Usenet newsgroups:
// MAGIC 
// MAGIC     alt.atheism
// MAGIC     comp.graphics
// MAGIC     comp.os.ms-windows.misc
// MAGIC     comp.sys.ibm.pc.hardware
// MAGIC     comp.sys.mac.hardware
// MAGIC     comp.windows.x
// MAGIC     misc.forsale
// MAGIC     rec.autos
// MAGIC     rec.motorcycles
// MAGIC     rec.sport.baseball
// MAGIC     rec.sport.hockey
// MAGIC     sci.crypt
// MAGIC     sci.electronics
// MAGIC     sci.med
// MAGIC     sci.space
// MAGIC     soc.religion.christian
// MAGIC     talk.politics.guns
// MAGIC     talk.politics.mideast
// MAGIC     talk.politics.misc
// MAGIC     talk.religion.misc
// MAGIC 
// MAGIC Some of the newsgroups seem pretty similar on first glance, such as *comp.sys.ibm.pc.hardware* and *comp.sys.mac.hardware*, which may affect our results.

// COMMAND ----------

// MAGIC %md ##Loading the Data and Data Cleaning
// MAGIC 
// MAGIC We will use the wget command to download the file, and read in the data using wholeTextFiles().

// COMMAND ----------

// MAGIC %sh wget http://kdd.ics.uci.edu/databases/20newsgroups/mini_newsgroups.tar.gz -O /tmp/newsgroups.tar.gz

// COMMAND ----------

// MAGIC %md Untar the file into the /tmp/ folder.

// COMMAND ----------

// MAGIC %sh tar xvfz /tmp/newsgroups.tar.gz -C /tmp/

// COMMAND ----------

// MAGIC %md The below cell takes about 10mins to run.

// COMMAND ----------

// MAGIC %fs cp -r file:/tmp/mini_newsgroups dbfs:/tmp/mini_newsgroups

// COMMAND ----------

// MAGIC %md
// MAGIC The `wholeTextFiles()` command will read in the entire directory of text files, and return a key-value pair of (filePath, fileContent).
// MAGIC 
// MAGIC As we do not need the file paths in this example, we will apply a map function to extract the file contents, and then convert everything to lowercase.

// COMMAND ----------

// Load text file, leave out file paths, convert all strings to lowercase
val corpus = sc.wholeTextFiles("/tmp/mini_newsgroups/*").map(_._2).map(_.toLowerCase())

// COMMAND ----------

// MAGIC %md Let's review a random document in the corpus.

// COMMAND ----------

corpus.takeSample(false, 1)

// COMMAND ----------

// MAGIC %md Note that the document begins with a header containing some metadata that we don't need, and we are only interested in the body of the document. We can do a bit of simple data cleaning here by removing the metadata of each document, which reduces the noise in our dataset. This is an important step as the accuracy of our models depend greatly on the quality of data used.

// COMMAND ----------

// Split document by double newlines, drop the first block, combine again as a string
val corpus_body = corpus.map(_.split("\\n\\n")).map(_.drop(1)).map(_.mkString(" "))

// COMMAND ----------

// Review first 5 documents with metadata removed
corpus_body.take(5)

// COMMAND ----------

// MAGIC %md To use the convenient [Feature extraction and transformation APIs](http://spark.apache.org/docs/latest/ml-features.html), we will convert our RDD into a DataFrame.
// MAGIC 
// MAGIC We will also create an ID for every document.

// COMMAND ----------

// Convert RDD to DF with ID for every document
val corpus_df = corpus_body.zipWithIndex.toDF("corpus", "id")

// COMMAND ----------

display(corpus_df)

// COMMAND ----------

// MAGIC %md ## Text Tokenization
// MAGIC 
// MAGIC We will use the `RegexTokenizer` to split each document into tokens. We can `setMinTokenLength()` here to indicate a minimum token length, and filter away all tokens that fall below the minimum.

// COMMAND ----------

import org.apache.spark.ml.feature.RegexTokenizer

// Set params for RegexTokenizer
val tokenizer = new RegexTokenizer()
  .setPattern("[\\W_]+")
  .setMinTokenLength(4) // Filter away tokens with length < 4
  .setInputCol("corpus")
  .setOutputCol("tokens")

// Tokenize document
val tokenized_df = tokenizer.transform(corpus_df)

// COMMAND ----------

display(tokenized_df.select("tokens"))

// COMMAND ----------

// MAGIC %md ## Remove Stopwords
// MAGIC 
// MAGIC We can easily remove stopwords using the `StopWordsRemover()`. If a list of stopwords is not provided, the `StopWordsRemover()` will use [this list of stopwords](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words) by default. You can use `getStopWords()` to see the list of stopwords that will be used.
// MAGIC 
// MAGIC In this example, we will specify a list of stopwords for the `StopWordsRemover()` to use. We do this so that we can add on to the list later on.

// COMMAND ----------

// MAGIC %sh wget http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words -O /tmp/stopwords

// COMMAND ----------

// MAGIC %fs cp file:/tmp/stopwords dbfs:/tmp/stopwords

// COMMAND ----------

// List of stopwords
val stopwords = sc.textFile("/tmp/stopwords").collect()

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

// Set params for StopWordsRemover
val remover = new StopWordsRemover()
  .setStopWords(stopwords) // This parameter is optional
  .setInputCol("tokens")
  .setOutputCol("filtered")

// Create new DF with Stopwords removed
val filtered_df = remover.transform(tokenized_df)

// COMMAND ----------

// MAGIC %md ## Vector of Token Counts
// MAGIC 
// MAGIC LDA takes in a vector of token counts as input. We can use the `CountVectorizer()` to easily convert our text documents into vectors of token counts.
// MAGIC 
// MAGIC The `CountVectorizer` will return (VocabSize, Array(Indexed Tokens), Array(Token Frequency))
// MAGIC 
// MAGIC Two handy parameters to note:
// MAGIC   - `setMinDF`: Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
// MAGIC   - `setMinTF`: Specifies the minimumn number of times a term has to appear in a document to be included in the vocabulary.

// COMMAND ----------

import org.apache.spark.ml.feature.CountVectorizer

// Set params for CountVectorizer
val vectorizer = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  .setVocabSize(10000)
  .setMinDF(5)
  .fit(filtered_df)

// COMMAND ----------

// Create vector of token counts
val countVectors = vectorizer.transform(filtered_df).select("id", "features")

// COMMAND ----------

// MAGIC %md To use the LDA algorithm in the MLlib library, we have to convert the DataFrame back into an RDD.

// COMMAND ----------

// Convert DF to RDD
import org.apache.spark.mllib.linalg.Vector

val lda_countVector = countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }

// COMMAND ----------

// format: Array(id, (VocabSize, Array(indexedTokens), Array(Token Frequency)))
lda_countVector.take(1)

// COMMAND ----------

// MAGIC %md ## Create LDA model with Online Variational Bayes
// MAGIC 
// MAGIC We will now set the parameters for LDA. We will use the `OnlineLDAOptimizer()` here, which implements Online Variational Bayes.
// MAGIC 
// MAGIC Choosing the number of topics for your LDA model requires a bit of domain knowledge. As we know that there are 20 unique newsgroups in our dataset, we will set numTopics to be 20.

// COMMAND ----------

val numTopics = 20

// COMMAND ----------

// MAGIC %md
// MAGIC We will set the parameters needed to build our LDA model. We can also `setMiniBatchFraction` for the `OnlineLDAOptimizer`, which sets the fraction of corpus sampled and used at each iteration. In this example, we will set this to 0.8.

// COMMAND ----------

import org.apache.spark.mllib.clustering.{LDA, OnlineLDAOptimizer}

// Set LDA params
val lda = new LDA()
  .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
  .setK(numTopics)
  .setMaxIterations(3)
  .setDocConcentration(-1) // use default values
  .setTopicConcentration(-1) // use default values

// COMMAND ----------

// MAGIC %md Create the LDA model with Online Variational Bayes.

// COMMAND ----------

val ldaModel = lda.run(lda_countVector)

// COMMAND ----------

// MAGIC %md Note that using the `OnlineLDAOptimizer` returns us a [LocalLDAModel](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.LocalLDAModel), which stores the inferred topics of your corpus.

// COMMAND ----------

// MAGIC %md ## Review Topics
// MAGIC 
// MAGIC We can now review the results of our LDA model. We will print out all 20 topics with their corresponding term probabilities.
// MAGIC 
// MAGIC Note that you will get slightly different results every time you run an LDA model since LDA includes some randomization.

// COMMAND ----------

// Review Results of LDA model with Online Variational Bayes
val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
val vocabList = vectorizer.vocabulary
val topics = topicIndices.map { case (terms, termWeights) =>
  terms.map(vocabList(_)).zip(termWeights)
}
println(s"$numTopics topics:")
topics.zipWithIndex.foreach { case (topic, i) =>
  println(s"TOPIC $i")
  topic.foreach { case (term, weight) => println(s"$term\t$weight") }
  println(s"==========")
}

// COMMAND ----------

// MAGIC %md Going through the results, you may notice that some of the topic words returned are actually stopwords that are specific to our dataset (for eg: "writes", "article"...). Let's try improving our model.

// COMMAND ----------

// MAGIC %md ## Model Tuning - Refilter Stopwords
// MAGIC 
// MAGIC We will try to improve the results of our model by identifying some stopwords that are specific to our dataset. We will filter these stopwords out and rerun our LDA model to see if we get better results.

// COMMAND ----------

val add_stopwords = Array("article", "writes", "entry", "date", "udel", "said", "tell", "think", "know", "just", "newsgroup", "line", "like", "does", "going", "make", "thanks")

// COMMAND ----------

// Combine newly identified stopwords to our exising list of stopwords
val new_stopwords = stopwords.union(add_stopwords)

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

// Set Params for StopWordsRemover with new_stopwords
val remover = new StopWordsRemover()
  .setStopWords(new_stopwords)
  .setInputCol("tokens")
  .setOutputCol("filtered")

// Create new df with new list of stopwords removed
val new_filtered_df = remover.transform(tokenized_df)

// COMMAND ----------

// Set Params for CountVectorizer
val vectorizer = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  .setVocabSize(10000)
  .setMinDF(5)
  .fit(new_filtered_df)

// Create new df of countVectors
val new_countVectors = vectorizer.transform(new_filtered_df).select("id", "features")

// COMMAND ----------

// Convert DF to RDD
val new_lda_countVector = new_countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }

// COMMAND ----------

// MAGIC %md We will also increase `MaxIterations` to 10 to see if we get better results.

// COMMAND ----------

// Set LDA parameters
val new_lda = new LDA()
  .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
  .setK(numTopics)
  .setMaxIterations(10)
  .setDocConcentration(-1) // use default values
  .setTopicConcentration(-1) // use default values

// COMMAND ----------

// Create LDA model with stopwords refiltered
val new_ldaModel = new_lda.run(new_lda_countVector)
// this will take a while to run!

// COMMAND ----------

val topicIndices = new_ldaModel.describeTopics(maxTermsPerTopic = 5)
val vocabList = vectorizer.vocabulary
val topics = topicIndices.map { case (terms, termWeights) =>
  terms.map(vocabList(_)).zip(termWeights)
}
println(s"$numTopics topics:")
topics.zipWithIndex.foreach { case (topic, i) =>
  println(s"TOPIC $i")
  topic.foreach { case (term, weight) => println(s"$term\t$weight") }
  println(s"==========")
}

// COMMAND ----------

// MAGIC %md We managed to get better results here. Your results will likely differ because of how LDA works and the randomness inherent in the model however we can see some strength in the general topics.

// COMMAND ----------

// MAGIC %md ## Create LDA model with Expectation Maximization
// MAGIC 
// MAGIC Let's try creating an LDA model with Expectation Maximization on the data that has been refiltered for additional stopwords. We will also increase `MaxIterations` here to 100 to see if that improves results.

// COMMAND ----------

// Set LDA parameters
val em_lda = new LDA()
  .setOptimizer("em")
  .setK(numTopics)
  .setMaxIterations(100)
  .setDocConcentration(-1) // use default values
  .setTopicConcentration(-1) // use default values

// COMMAND ----------

val em_ldaModel = em_lda.run(new_lda_countVector)

// COMMAND ----------

// MAGIC %md
// MAGIC Note that the `EMLDAOptimizer` produces a `DistributedLDAModel`, which stores not only the inferred topics but also the full training corpus and topic distributions for each document in the training corpus.

// COMMAND ----------

val topicIndices = em_ldaModel.describeTopics(maxTermsPerTopic = 5)
val vocabList = vectorizer.vocabulary
val topics = topicIndices.map { case (terms, termWeights) =>
  terms.map(vocabList(_)).zip(termWeights)
}
println(s"$numTopics topics:")
topics.zipWithIndex.foreach { case (topic, i) =>
  println(s"TOPIC $i")
  topic.foreach { case (term, weight) => println(s"$term\t$weight") }
  println(s"==========")
}

// COMMAND ----------

// MAGIC %md We managed to get better results here. Your results will likely differ because of how LDA works and the randomness inherent in the model however we can see some improvements in the topics.

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC To improve our results further, we could employ some of the below methods:
// MAGIC - Refilter data for additional data-specific stopwords
// MAGIC - Use Stemming or Lemmatization to preprocess data
// MAGIC - Experiment with a smaller number of topics, since some of these topics in the 20 Newsgroups are pretty similar
// MAGIC - Increase model's MaxIterations

// COMMAND ----------

// MAGIC %md ## Visualize Results
// MAGIC 
// MAGIC We will try visualizing the results obtained from the EM LDA model with a d3 bubble chart.

// COMMAND ----------

// Zip topic terms with topic IDs
val termArray = topics.zipWithIndex

// COMMAND ----------

// Transform data into the form (term, probability, topicId)
val termRDD = sc.parallelize(termArray)
val termRDD2 =termRDD.flatMap( (x: (Array[(String, Double)], Int)) => {
  val arrayOfTuple = x._1
  val topicId = x._2
  arrayOfTuple.map(el => (el._1, el._2, topicId))
})

// COMMAND ----------

// Create DF with proper column names
val termDF = termRDD2.toDF.withColumnRenamed("_1", "term").withColumnRenamed("_2", "probability").withColumnRenamed("_3", "topicId")

// COMMAND ----------

display(termDF)

// COMMAND ----------

// MAGIC %md We will convert the DataFrame into a JSON format, which will be passed into d3.

// COMMAND ----------

// Create JSON data
val rawJson = termDF.toJSON.collect().mkString(",\n")

// COMMAND ----------

// MAGIC %md We are now ready to use D3 on the rawJson data.

// COMMAND ----------

displayHTML(s"""
<!DOCTYPE html>
<meta charset="utf-8">
<style>

circle {
  fill: rgb(31, 119, 180);
  fill-opacity: 0.5;
  stroke: rgb(31, 119, 180);
  stroke-width: 1px;
}

.leaf circle {
  fill: #ff7f0e;
  fill-opacity: 1;
}

text {
  font: 14px sans-serif;
}

</style>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script>

var json = {
 "name": "data",
 "children": [
  {
     "name": "topics",
     "children": [
      ${rawJson}
     ]
    }
   ]
};

var r = 1500,
    format = d3.format(",d"),
    fill = d3.scale.category20c();

var bubble = d3.layout.pack()
    .sort(null)
    .size([r, r])
    .padding(1.5);

var vis = d3.select("body").append("svg")
    .attr("width", r)
    .attr("height", r)
    .attr("class", "bubble");

  
var node = vis.selectAll("g.node")
    .data(bubble.nodes(classes(json))
    .filter(function(d) { return !d.children; }))
    .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
    color = d3.scale.category20();
  
  node.append("title")
      .text(function(d) { return d.className + ": " + format(d.value); });

  node.append("circle")
      .attr("r", function(d) { return d.r; })
      .style("fill", function(d) {return color(d.topicName);});

var text = node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", ".3em")
    .text(function(d) { return d.className.substring(0, d.r / 3)});
  
  text.append("tspan")
      .attr("dy", "1.2em")
      .attr("x", 0)
      .text(function(d) {return Math.ceil(d.value * 10000) /10000; });

// Returns a flattened hierarchy containing all leaf nodes under the root.
function classes(root) {
  var classes = [];

  function recurse(term, node) {
    if (node.children) node.children.forEach(function(child) { recurse(node.term, child); });
    else classes.push({topicName: node.topicId, className: node.term, value: node.probability});
  }

  recurse(null, root);
  return {children: classes};
}
</script>
""")
