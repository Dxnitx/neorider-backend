from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base


class RideSummary(Base):
    __tablename__ = "ride_summaries"

    id = Column(Integer, primary_key=True, index=True)
    ride_id = Column(Integer, ForeignKey("rides.id"), nullable=False)
    posture_score = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    accident_risk_count = Column(Integer, default=0)
    total_alerts = Column(Integer, default=0)
    summary_text = Column(String(255), nullable=True)
    recommendation = Column(String(255), nullable=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())