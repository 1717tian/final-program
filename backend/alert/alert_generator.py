# backend/alert/alert_generator.py
from backend.alert.alert_rules import pm25_alert_level

class AlertGenerator:
    """
    生成空气质量预警
    """
    def __init__(self):
        pass

    def generate_alert(self, pm25_value):
        level, color = pm25_alert_level(pm25_value)
        return {"pm25": pm25_value, "level": level, "color": color}