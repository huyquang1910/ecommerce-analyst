# Databricks notebook source
import numpy as np 
import pandas as pd # data processing
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import count, sum,avg

# COMMAND ----------

spark = SparkSession.builder.appName("Brazil E-commerce analyst").getOrCreate()


# COMMAND ----------

spark

# COMMAND ----------

sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ---Prepare dataset from kaggle

# COMMAND ----------

categories = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/product_category_name_translation.csv")
sellers = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_sellers_dataset.csv")
products = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_products_dataset.csv")
reviews = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_order_reviews_dataset.csv")
orders = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_orders_dataset.csv")
payments = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_order_payments_dataset.csv")
order_items = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_order_items_dataset.csv")
customers = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_customers_dataset.csv")
geolocations = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/huyq1910@gmail.com/olist_geolocation_dataset.csv")

# COMMAND ----------

orders.groupBy("order_status").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC --- Analysis and Visualize data

# COMMAND ----------

# Create temp table
categories.createOrReplaceTempView("categories")
sellers.createOrReplaceTempView("sellers")
products.createOrReplaceTempView("products")
reviews.createOrReplaceTempView("reviews")
orders.createOrReplaceTempView("orders")
payments.createOrReplaceTempView("payments")
order_items.createOrReplaceTempView("order_items")
customers.createOrReplaceTempView("customers")
geolocations.createOrReplaceTempView("geolocations")

# Show all temp table
spark.sql("SHOW TABLES").show()

# COMMAND ----------

# 1.1 Phân bố số lượng người dùng ở mỗi bang
spark.sql("select customer_state, count(*) as customer_count from customers group by customer_state order by customer_count desc").show()

# COMMAND ----------

#Use dataframe
customer_state_distribution=customers.groupBy("customer_state").agg(count('*').alias("customer_count")).orderBy("customer_count", ascending=False)
customer_state_distribution.show()

# COMMAND ----------

print(type(customers))

# COMMAND ----------

#convert to pandas dataframe
customer_state_distribution_pd = customer_state_distribution.toPandas()
# Define values, label
labels = customer_state_distribution_pd["customer_state"]
sizes = customer_state_distribution_pd["customer_count"]

# Pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    sizes,
    labels=labels,
    startangle=90,      # bat dau tu 90 do
    textprops={'fontsize': 10},  # 
)
plt.title("User Distribution by State (Percentage)", fontsize=14)
plt.tight_layout()
plt.show()


# COMMAND ----------

# 1.2 Phân bổ trạng thái đơn hàng
spark.sql("select order_status, count(*) as order_count from orders group by order_status order by order_count desc").show()

# COMMAND ----------

#Use dataframe
order_status_distribution=orders.groupBy("order_status").agg(count("*").alias('order_count')).orderBy("order_count",ascending=False)
order_status_distribution.show()

# COMMAND ----------

order_status_distribution_pd = order_status_distribution.toPandas()
labels = order_status_distribution_pd["order_status"]
sizes = order_status_distribution_pd["order_count"]
# Chart
plt.figure(figsize=(12, 6))
plt.bar(order_status_distribution_pd["order_status"], order_status_distribution_pd["order_count"], color='lightblue')
plt.title("Order Status Distribution", fontsize=14)
plt.xlabel("Order Status", fontsize=12)
plt.ylabel("Order Count", fontsize=12)
plt.xticks(rotation=15)  # Xoay label theo truc ngang
plt.tight_layout()
plt.show()

# COMMAND ----------

#1.3 Đếm số lượng đơn hàng và tổng doanh số theo tiểu bang
orders.printSchema()
order_items.printSchema()
customers.printSchema()

# COMMAND ----------

spark.sql("""SELECT 
    customer_state, count("order_id") as order_count, sum(CAST(price AS DECIMAL(10, 2))) as total_sales
FROM 
    orders
JOIN 
    order_items ON orders.order_id = order_items.order_id
JOIN 
    customers ON orders.customer_id = customers.customer_id
GROUP BY customer_state order by total_sales desc""").show()


# COMMAND ----------

#using data frame
from pyspark.sql.functions import *
round(sum(col("price").cast("double")), 2)
order_data = orders.join(order_items, "order_id").join(customers, "customer_id")
state_sales = order_data.groupBy("customer_state").agg(
    count("order_id").alias("order_count"),
    sum("price").alias("total_sales")
).orderBy("total_sales", ascending=False)
state_sales.show()

# COMMAND ----------

#1.4 Thống kê tổng doanh số và xếp hạng trung bình theo danh mục sản phẩm
order_items.printSchema()
products.printSchema()
reviews.printSchema()

# COMMAND ----------

spark.sql("""
          SELECT product_category_name, round(sum(price),2) as total_sales, round(avg(review_score),2) as avg_review_score FROM order_items JOIN products ON order_items.product_id=products.product_id JOIN reviews on reviews.order_id= order_items.order_id GROUP BY product_category_name ORDER BY total_sales desc""").show()

# COMMAND ----------

#using dataframe
# join 3 table order_items、products,reviews
product_data = order_items.join(products, "product_id").join(reviews, "order_id")
category_performance = product_data.groupBy("product_category_name").agg(
    round(sum("price"), 2).alias("total_sales"),  
    round(avg("review_score"), 2).alias("avg_review_score")  
).orderBy("total_sales", ascending=False)
category_performance.show()


# COMMAND ----------

#Top 20 danh mục sp có doanh thu cao nhất
category_performance.createOrReplaceTempView("category_performance")
spark.sql("""SELECT product_category_name, total_sales from category_performance ORDER BY total_sales desc limit 20
          """).show()

# COMMAND ----------

top_sales_categories=category_performance.select("product_category_name","total_sales").orderBy("total_sales",ascending=False).limit(20)
top_sales_categories.show()

# COMMAND ----------

top_sales_categories_pd=top_sales_categories.toPandas()
plt.figure(figsize=(12, 6))
plt.bar(top_sales_categories_pd["product_category_name"], top_sales_categories_pd["total_sales"], color='skyblue', edgecolor='black')
plt.xticks(rotation=40, fontsize=8, ha='right')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.title('Top 20 Categories by Sales')
plt.tight_layout()
plt.show()

# COMMAND ----------

#Top 20 danh mục sp score cao nhất
spark.sql("""SELECT product_category_name, avg_review_score from category_performance ORDER BY avg_review_score desc limit 20
          """).show()

# COMMAND ----------

#use dataframe
top_avg_review_score_categories=category_performance.select("product_category_name","avg_review_score").orderBy("avg_review_score",ascending=False).limit(20)
top_avg_review_score_categories.show()

# COMMAND ----------

top_avg_review_score_categories_pd=top_avg_review_score_categories.toPandas()
plt.figure(figsize=(12, 6))
plt.bar(top_avg_review_score_categories_pd["product_category_name"], top_avg_review_score_categories_pd["avg_review_score"], color='skyblue', edgecolor='black')
plt.xticks(rotation=40, fontsize=8, ha='right')
plt.xlabel('Product Category')
plt.ylabel('Total avg review score categories')
plt.title('Top 20 Categories by review score')
plt.tight_layout()
plt.show()

# COMMAND ----------

#1.5 Đếm số lượng đơn hàng và tổng số tiền đặt hàng của từng khách hàng
orders.printSchema()
order_items.printSchema()
customers.printSchema()

# COMMAND ----------

customer_orders = orders.join(customers, "customer_id")
customer_data = customer_orders.join(order_items, "order_id")

# COMMAND ----------

customer_behavior = customer_data.groupBy("customer_unique_id").agg(
    count("order_id").alias("order_count"),
    round(sum("price"),1).alias("sum_order_value")
).orderBy("order_count", ascending=False)
customer_behavior.show()

# COMMAND ----------

#1.6 Thống kê số lượng đặt hàng và số lượng đặt hàng trung bình của từng khách hàng

# COMMAND ----------

customer_behavior_2 = customer_data.groupBy("customer_id").agg(
    count("order_id").alias("order_count"),
    round(avg("price"),1).alias("avg_order_value")
).orderBy("order_count", ascending=False)
customer_behavior_2.show()

# COMMAND ----------

customer_geo_data = customers.join(
    geolocations,
    customers.customer_zip_code_prefix == geolocations.geolocation_zip_code_prefix
).select(
    "customer_id", "customer_zip_code_prefix", 
    "geolocation_lat", "geolocation_lng", 
    "customer_city", "customer_state"
)

# COMMAND ----------

customer_behavior_geo = customer_behavior_2.join(
    customer_geo_data, "customer_id"
)
customer_behavior_geo.show()

# COMMAND ----------

#SL đơn hàng theo vị trí địa lý
geo_distribution = customer_behavior_geo.groupBy(
    "geolocation_lat", "geolocation_lng"
).agg(
    sum("order_count").alias("total_orders"),
).orderBy("total_orders", ascending=False)

geo_distribution.show()

# COMMAND ----------

geo_distribution.count()

# COMMAND ----------

geo_distribution_pd = geo_distribution.toPandas()


# COMMAND ----------

import folium
from folium.plugins import HeatMap
heat_data = geo_distribution_pd[["geolocation_lat", "geolocation_lng", "total_orders"]].values.tolist()
heat_data_small = heat_data[:400000]
m = folium.Map(location=[-14.2350, -51.9253], zoom_start=5) 
HeatMap(heat_data_small, radius=10).add_to(m)


# COMMAND ----------

m

# COMMAND ----------

#1.8. Xu hướng về tổng doanh thu hàng năm và số lượng đặt hàng trung bình
monthly_sales = orders.join(order_items, "order_id").select(
    year("order_purchase_timestamp").alias("year"),
    month("order_purchase_timestamp").alias("month"),
    "price"
).groupBy(
    "year", "month"  
).agg(
    sum("price").alias("total_sales")
).orderBy(
    "year", "month" 
)

monthly_sales_pd = monthly_sales.withColumn(
    "year_month", concat_ws("-", col("year"), col("month"))
).select(
    "year_month", "total_sales"
).toPandas()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales_pd["year_month"], 
    monthly_sales_pd["total_sales"], 
    marker="o", linestyle="-", color="blue"
)
plt.xticks(rotation=45)
plt.title("Monthly Total Sales Trend", fontsize=14)
plt.xlabel("Year-Month", fontsize=12)
plt.ylabel("Total Sales (BRL)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

#1.9 Phân tích người bán
seller_sales = order_items.groupBy("seller_id").agg(
    sum("price").alias("total_sales")
).orderBy("total_sales", ascending=False).limit(20)
seller_sales_pd = seller_sales.toPandas()
print("Top 20 Sellers by Total Sales:")
print(seller_sales_pd)

# COMMAND ----------


seller_order_counts = order_items.groupBy("seller_id").agg(
    count("order_id").alias("order_count")
).orderBy("order_count", ascending=False).limit(20)

seller_order_counts_pd = seller_order_counts.toPandas()

print("Top 20 Sellers by Order Count:")
print(seller_order_counts_pd)

# COMMAND ----------

seller_state_distribution = sellers.groupBy("seller_state").count()
seller_state_distribution_pd = seller_state_distribution.toPandas()
print("Seller Distribution by State:")
print(seller_state_distribution_pd)
