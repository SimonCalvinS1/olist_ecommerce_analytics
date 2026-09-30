# Olist Supabase ETL

# Dataset Details
This project uses Olist's publicly available CSV datasets (Brazilian E-Commerce Public Dataset by Olist). Accessible through [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce].

This Python script reads the Olist CSVs, calculates four useful summaries, and replaces those tables in Supabase PostgreSQL using an ETL pipeline:

- `analytics.monthly_sales`: delivered orders, item count, item revenue, and freight by month.
- `analytics.category_sales`: delivered orders, item count, and item revenue by product category.
- `analytics.payment_summary`: payment count and total value by payment type.
- `analytics.review_summary`: number of reviews by score.

It reads only the files needed for these summaries; the large geolocation file is not loaded.

## Setup

The pipeline uses `SUPABASE_PROJECT_REF`. Set `SUPABASE_CONNECTION_STRING` to the PostgreSQL URI from Supabase **Project Settings > Database**. `.env` is ignored by Git. Install dependencies from this folder:

```powershell
pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
python real_world_dataset_olist_analytics/olist_elt_pipeline/pipeline.py
```

The script reads CSVs from the sibling `olist_data` folder. Each run replaces only the four tables listed above in the `analytics` schema. Query them in the Supabase SQL editor, for example:

```sql
SELECT *
FROM analytics.monthly_sales
ORDER BY sales_month;
```
