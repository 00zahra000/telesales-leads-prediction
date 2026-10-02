"""Create the PostgreSQL engine from environment variables."""

import os

from sqlalchemy import URL, create_engine


def create_db_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(url, connect_args={"connect_timeout": 10})
