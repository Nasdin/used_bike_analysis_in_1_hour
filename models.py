from typing import Optional

from sqlmodel import SQLModel, Field


class Brand(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, nullable=False)


class Model(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    brand_id: Optional[int] = Field(default=None, foreign_key="brand.id")
    name: str = Field(nullable=False)
