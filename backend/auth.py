from passlib.context import CryptContext
from sqlalchemy.orm import Session
from db import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def register_user(db: Session, username: str, password: str, role: str = "user"):
    user_exists = db.query(User).filter(User.username == username).first()
    if user_exists:
        return {"success": False, "message": "用户名已存在"}

    hashed_pw = get_password_hash(password)
    new_user = User(username=username, password=hashed_pw, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"success": True, "message": "注册成功"}


def login_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password):
        return {"success": False, "message": "用户名或密码错误"}
    return {"success": True, "username": user.username, "role": user.role}