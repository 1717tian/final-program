import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { login } from "../api/backend";

export default function Login() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: "",
    password: "",
  });

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!form.username.trim()) {
      alert("请输入用户名");
      return;
    }

    if (!form.password.trim()) {
      alert("请输入密码");
      return;
    }

    const result = await login({
      username: form.username.trim(),
      password: form.password,
    });

    if (result.success) {
      localStorage.setItem("isLogin", "true");
      localStorage.setItem("username", result.username || form.username.trim());
      localStorage.setItem("role", result.role || "user");

      if (result.role === "admin") {
        navigate("/admin");
      } else {
        navigate("/prediction");
      }
    } else {
      alert(result.message || "登录失败");
    }
  };

  return (
    <div className="home-container">
      <Navbar />

      <main className="auth-page">
        <form className="login-box" onSubmit={handleLogin}>
          <h2>用户登录</h2>

          <input
            type="text"
            name="username"
            placeholder="用户名"
            value={form.username}
            onChange={handleChange}
          />

          <input
            type="password"
            name="password"
            placeholder="密码"
            value={form.password}
            onChange={handleChange}
          />

          <button type="submit">登录</button>

          <p className="form-tip">
            没有账号？
            <Link to="/register"> 注册 </Link>
            <Link to="/admin-login">切换到管理员登录</Link>
          </p>
        </form>
      </main>

      <Footer />
    </div>
  );
}