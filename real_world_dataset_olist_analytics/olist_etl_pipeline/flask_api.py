from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache

from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from pipeline import get_database_url


app = Flask(__name__)
ANALYTICS_TABLES = (
    "monthly_sales",
    "category_sales",
    "payment_summary",
    "review_summary",
)
MAX_LIMIT = 5000


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


def json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


@app.get("/")
def index():
    return jsonify(
        service="Olist analytics API",
        endpoints={
            "analytics": "/api/analytics",
            "tables": [f"/api/analytics/{table}" for table in ANALYTICS_TABLES],
        },
    )


@app.get("/api/analytics")
def list_analytics():
    return jsonify(tables=list(ANALYTICS_TABLES))


@app.get("/api/analytics/<table_name>")
def get_analytics(table_name: str):
    if table_name not in ANALYTICS_TABLES:
        return jsonify(error="Unknown analytics table."), 404

    raw_limit = request.args.get("limit")
    try:
        limit = 1000 if raw_limit is None else int(raw_limit)
    except ValueError:
        return jsonify(error="limit must be a positive integer."), 400
    if limit < 1:
        return jsonify(error="limit must be a positive integer."), 400
    limit = min(limit, MAX_LIMIT)

    try:
        with get_engine().connect() as connection:
            result = connection.execute(
                text(f"SELECT * FROM analytics.{table_name} LIMIT :limit"),
                {"limit": limit},
            )
            rows = [
                {column: json_value(value) for column, value in row.items()}
                for row in result.mappings()
            ]
    except RuntimeError as error:
        return jsonify(error=str(error)), 503
    except SQLAlchemyError:
        app.logger.exception("Failed to read analytics table %s", table_name)
        return jsonify(
            error="Unable to read analytics data. Check the database settings and run pipeline.py."
        ), 503

    return jsonify(table=table_name, count=len(rows), data=rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)