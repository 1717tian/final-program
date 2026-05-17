from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import yaml

# 读取数据库配置
with open("backend/config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_conf = config['database']
DB_URL = f"postgresql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf['port']}/{db_conf['dbname']}"

engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """
    自动检测 pm25_data 表是否存在，
    不存在则创建表结构
    """
    print("正在检查数据库表结构...")
    Base.metadata.create_all(engine)  # 如果表不存在会自动创建
    print("数据库初始化完成！")