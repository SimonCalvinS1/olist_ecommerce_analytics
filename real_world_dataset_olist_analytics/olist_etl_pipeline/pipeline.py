from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url


PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR.parent / "olist_data"


def build_summaries(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    items = pd.read_csv(
        data_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price", "freight_value"],
    )
    delivered = orders.loc[orders["order_status"] == "delivered"]
    delivered_items = items.merge(delivered, on="order_id", how="inner")
    delivered_items["sales_month"] = (
        delivered_items["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_sales = (
        delivered_items.groupby("sales_month", as_index=False)
        .agg(
            delivered_orders=("order_id", "nunique"),
            items_sold=("order_id", "size"),
            item_revenue=("price", "sum"),
            freight_revenue=("freight_value", "sum"),
        )
    )

    products = pd.read_csv(
        data_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    translations = pd.read_csv(data_dir / "product_category_name_translation.csv")
    category_items = delivered_items.merge(products, on="product_id", how="left").merge(
        translations, on="product_category_name", how="left"
    )
    category_items["category"] = category_items[
        "product_category_name_english"
    ].fillna(category_items["product_category_name"]).fillna("unknown")
    category_sales = (
        category_items.groupby("category", as_index=False)
        .agg(
            delivered_orders=("order_id", "nunique"),
            items_sold=("order_id", "size"),
            item_revenue=("price", "sum"),
        )
        .sort_values("item_revenue", ascending=False)
    )

    payments = pd.read_csv(
        data_dir / "olist_order_payments_dataset.csv",
        usecols=["payment_type", "payment_value"],
    )
    payment_summary = (
        payments.groupby("payment_type", as_index=False)
        .agg(
            payment_count=("payment_value", "size"),
            total_payment_value=("payment_value", "sum"),
        )
        .sort_values("total_payment_value", ascending=False)
    )

    reviews = pd.read_csv(
        data_dir / "olist_order_reviews_dataset.csv",
        usecols=["review_score"],
    )
    review_summary = (
        reviews.groupby("review_score", as_index=False)
        .agg(review_count=("review_score", "size"))
        .sort_values("review_score")
    )

    return {
        "monthly_sales": monthly_sales,
        "category_sales": category_sales,
        "payment_summary": payment_summary,
        "review_summary": review_summary,
    }


def main() -> None:
    load_dotenv(PIPELINE_DIR.parent / ".env")
    project_ref = os.getenv("SUPABASE_PROJECT_REF") or os.getenv("SUPABASE_PROJECT_NAME")
    password = os.getenv("SUPABASE_PASSWORD")
    if project_ref and password and re.fullmatch(r"[a-z0-9]{20}", project_ref):
        url = URL.create(
            "postgresql+psycopg",
            username="postgres",
            password=password,
            host=f"db.{project_ref}.supabase.co",
            port=5432,
            database="postgres",
            query={"sslmode": "require"},
        )
    else:
        connection_string = (
            os.getenv("SUPABASE_CONNECTION_STRING")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DATABASE_URL")
        )
        if not connection_string:
            raise RuntimeError(
                "Set SUPABASE_CONNECTION_STRING or SUPABASE_PROJECT_REF and "
                "SUPABASE_PASSWORD in real_world_dataset_olist_analytics/.env."
            )
        url = make_url(connection_string)
        if url.drivername in {"postgres", "postgresql"}:
            url = url.set(drivername="postgresql+psycopg")

    summaries = build_summaries()
    with create_engine(url).begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))
        for table_name, summary in summaries.items():
            summary.to_sql(
                table_name,
                connection,
                schema="analytics",
                if_exists="replace",
                index=False,
                method="multi",
            )
            print(f"Wrote analytics.{table_name}: {len(summary):,} rows")


if __name__ == "__main__":
    main()