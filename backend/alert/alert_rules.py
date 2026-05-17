# backend/alert/alert_rules.py
def pm25_alert_level(value):
    """
    根据PM2.5数值返回等级和颜色
    """
    if value <= 35:
        return "优", "green"
    elif value <= 75:
        return "良", "yellow"
    elif value <= 115:
        return "轻度污染", "orange"
    elif value <= 150:
        return "中度污染", "red"
    else:
        return "重度污染", "purple"