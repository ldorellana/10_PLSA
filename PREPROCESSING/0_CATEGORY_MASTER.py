# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # CATEGORY MASTER REVIEW

# COMMAND ----------

ls ../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/

# COMMAND ----------

import pandas as pd

df_new = pd.read_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/new_cat_data.csv', 
                     dtype='str',  
                     usecols=[9,10,7,8],
                     skiprows=2
                    )


df_old = pd.read_csv('../../dbfs/FileStore/tables/10_PLSA/DATA/ANALYSIS/_product_master_商品マスタ.csv', 
                     dtype='str', 
                     encoding='cp932', 
                     usecols=[0,2,6,7]
                    )

# COMMAND ----------

df_old.nunique()

# COMMAND ----------

df_new.nunique()

# COMMAND ----------

df_new.columns

# COMMAND ----------

df_new.columns = ['', 'CATEGORY　NAME', '商品ｺｰﾄﾞ', '商品名']
