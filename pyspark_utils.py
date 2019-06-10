import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.session import SparkSession

import numpy as np
import pandas as pd
import scipy

import pyspark.sql.functions as psf
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType,IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH, IDF, Tokenizer, Normalizer, CountVectorizer, StringIndexer

from pyspark.mllib.feature import ElementwiseProduct
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix

ss = SparkSession.builder.config("spark.sql.hive.verifyPartitionPath", "false").enableHiveSupport().getOrCreate()
sc = ss.sparkContext
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

sc._conf.set("spark.sql.hive.verifyPartitionPath", "false")

def task_serialization(df):
    df_sch = df.schema
    df_rdd = df.rdd
    return sqlContext.createDataFrame(df_rdd, schema=df_sch)

def union_all(dfs):
    if len(dfs) > 1:
        return dfs[0].unionAll(union_all(dfs[1:]))
    else:
        return dfs[0]



# Crete Bins based on spark column

def bins_fn(x, lo, high, period):
    ix = 0
    try:
        x = int(x)
    except:
        return "{},null".format(0)
    if  x == None: return "{},null".format(0)
    cat = "{},{}<".format(1,lo)
    if x < lo: return cat
    ix = 2
    while lo < high:
        if x >= lo and x < min(high, lo+period):
            return "{},{}-{}".format(ix,lo,  lo+period)
        ix+=1
        lo+= period
    if x >= high:
        return "{},>={}".format(ix,high)
    return "0,null"

def udf_bin(lo, high, size):
    return psf.udf(lambda l: bins_fn(l, lo, high, size))

def get_bins_with_freq(spark_df, column, bin_lo, bin_high, bin_size, bin_column="bins"):
    '''
    create bins based on a column
    spark_df : spark dataframe
    column : field in consideration
    bin_lo : starting value for the bins
    bin_size : size of each bin
    bin_column : name of the bin column
    '''
    binned_df = spark_df.withColumn('tmpbins', udf_bin( bin_lo, bin_high, bin_size)(psf.col(column)) )
    #print (binned_df.show(10))
    binned_freq = binned_df.groupBy(['tmpbins']).agg(psf.count(psf.lit(1) ).alias('count') )
    binned_freq = binned_freq.orderBy(psf.split(psf.col('tmpbins'),',').getItem(0).cast(IntegerType()) )
    binned_freq = binned_freq.withColumn(bin_column, psf.split(psf.col('tmpbins'),',').getItem(1))
    binned_freq =  binned_freq.drop("tmpbins")
    return binned_freq


#Find the kth largest value in th spark df / Find Percent Point

def get_kth_largest(df, field, k, maxm, debug=False):
    '''
    finds the kth largest value of the field in sparkdataframe
    without using window function
    df: spark dataframe
    field: the field in consideration
    k : value of K
    maxm : value greater tha of equal to maxium possible value of the field
    '''
    lo = 0
    hi = maxm
    it = 1
    if debug:
        print("JOB_LOG", "bin search", "ix", field, "position")
    pv_grtr = -10000
    while True:
        curval = float(hi + lo)/ 2
        grtr_count = int(df.filter(psf.col(field) >= curval).count())
        if debug:
            print("JOB_LOG", "bin search", it, curval, grtr_count)
        if pv_grtr == grtr_count and grtr_count!=0:
            return df.filter(psf.col(field) >= curval).select(field).orderBy(field).limit(1).collect()[0][field]

        pv_grtr = grtr_count
        if k < grtr_count:
            lo = curval
        else:
            hi = curval
        it += 1
     
        
def find_percent_point(df, field, per, maxm=100000, order="desc", debug=False):
    per = float(per) / 100
    total = df.count()
    k = int(int(total) * per)
    if order == "asc":
        k = total - k
    if debug:
        print("JOB_LOG", " {} th position in descending order of {} ".format(k, field))
    return get_kth_largest(df, field, k, maxm, debug)



#barplot multiple columns of a data frame in same plot

import seaborn as sns
import matplotlib.pyplot as plt
def plot_multiple_cols(df, indexcol):
    df = df.melt(indexcol, var_name='cols',  value_name='vals')
    g = sns.barplot(x='vals', y=indexcol, hue='cols', data=df)


#evalualte a model from predictions and labels

def model_evaluation(prediction_label_df):
    '''
    used to find precision recall of the predictions given the prediction_label_df
    with columns 'prediction' and 'label'
    '''
    prediction_label_df = prediction_label_df.withColumn('label_int',
                                                         psf.col('label').cast(IntegerType())).drop(
        'label').withColumnRenamed('label_int',
                                   'label')

    prediction_label_df = prediction_label_df.withColumn('prediction_int',
                                                         psf.col('prediction').cast(IntegerType())).drop(
        'prediction').withColumnRenamed('prediction_int',
                                        'prediction')

    from pyspark.mllib.evaluation import MulticlassMetrics
    predictions_and_labels = prediction_label_df
    metrics = MulticlassMetrics(predictions_and_labels.rdd.map(lambda row: (float(row.prediction), float(row.label))))
    classes = [row.label for row in prediction_label_df.select('label').distinct().take(100)]
    print("classes", classes)
    prediction_label_df.cache()
    correct_count = prediction_label_df.filter(prediction_label_df.label == prediction_label_df.prediction).count()
    accuracy = (correct_count * 100) / prediction_label_df.count()
    conf_dict = {}
    tmpls = []
    for cl in classes:
        conf_dict[cl] = {}
        conf_dict[cl]['recall'] = metrics.recall(int(cl))
        conf_dict[cl]['precision'] = metrics.precision(int(cl))
        conf_dict[cl]['f1-score'] = 2 * (conf_dict[cl]['recall'] * conf_dict[cl]['precision']) / (
        conf_dict[cl]['precision'] + conf_dict[cl]['recall'])
        tmpls.append([cl, conf_dict[cl]['precision'], conf_dict[cl]['recall'], conf_dict[cl]['f1-score']])
    print("Accuracy:", round(
        float(prediction_label_df.filter(prediction_label_df.prediction == prediction_label_df.label).count()) / float(
            prediction_label_df.count()), 4))
    print(pd.DataFrame(tmpls, columns=['column', 'precision', 'recall', 'f1-score']).head(100))
    prediction_label_df.unpersist()
    
    
#very fast connected component finder
def find_connected_components(sc, edges, n_dist = 5):
    
    def cc_find_root(group, found_new_pair):
        node, values = group
        
        #find the smalled vertex id in the current adjecency list 
        smallest = min(values  + [node])
        
        
        emit = []
        #if smallest is not the node 
        #emit the pairs with smallest for each node in the group
        if smallest < node:
            emit.append((node, smallest))
            for vi in values:
                if vi != smallest:
                    emit.append((vi, smallest))
                    found_new_pair.add(1)
        return emit
    
    
    while True:
        edges.persist()
        found_new_pair = sc.accumulator(0)
        print ("JOB_LOG", "  ITERATION : ",n_dist )
        
        #ccf map
        s1 = edges.flatMap(lambda x : [(x[0], x[1]), (x[1], x[0])])
        #s1.take(1)
        
        #ccf reduce
        s2 = s1.groupByKey().map(lambda (x,y): (x, list(y)))
        #print s2.take(1)
        
        s3 = s2.flatMap(lambda x : cc_find_root(x, found_new_pair))
        s3.cache()
        print "JOB_LOG","total relations count:", s3.count() #necessary to perform action else accumulator will not update
        
        
        #ccf deduce
        edges.unpersist()
        edges = s3.distinct() 
        
        print "JOB_LOG","New pairs found :",found_new_pair.value
        if found_new_pair.value == 0 or n_dist == 0:
            break
        n_dist -= 1
    
    return edges.map(lambda x: (x[1], x[0])).groupByKey().map(lambda (x,y) : (x, list(y)))