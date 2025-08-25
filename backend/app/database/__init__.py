"""Database package for Auto Analytics Platform."""

from .mongodb import mongodb, get_mongodb, init_database, close_database
from .models import (
    DatasetDocument,
    AnalysisDocument,
    MLModelDocument,
    ReportDocument,
    ChatSessionDocument,
    UserDocument,
    RwandaProvinceDocument,
    RwandaDistrictDocument
)

__all__ = [
    "mongodb",
    "get_mongodb",
    "init_database",
    "close_database",
    "DatasetDocument",
    "AnalysisDocument", 
    "MLModelDocument",
    "ReportDocument",
    "ChatSessionDocument",
    "UserDocument",
    "RwandaProvinceDocument",
    "RwandaDistrictDocument"
]
