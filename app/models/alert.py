from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    ride_id = Column(Integer, ForeignKey("rides.id"), nullable=False)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    message = Column(String(255), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())