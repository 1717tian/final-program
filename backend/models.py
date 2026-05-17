from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PM25Data(Base):
    __tablename__ = "pm25_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_name = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    pm25_value = Column(Float)
    predicted = Column(Float)
    alert_level = Column(String(20))