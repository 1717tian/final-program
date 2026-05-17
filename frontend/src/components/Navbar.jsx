import React from "react";

export default function Navbar() {
  return (
    <div className="navbar">
      <div className="logo">PM2.5监控预警平台</div>
      <div className="nav-items">
        <a href="/">首页</a>
        <a href="/prediction">预测</a>
        <a href="/login">登录</a>
        <a href="/admin">管理员</a>
      </div>
    </div>
  );
}