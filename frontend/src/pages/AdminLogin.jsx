import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { adminLogin } from "../api/backend";

export default function AdminLogin() {
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

  const handleAdminLogin = async (e) => {
    e.preventDefault();

    if (!form.username.trim()) {
      alert("请输入管理员用户名");
      return;
    }

    if (!form.password.trim()) {
      alert("请输入管理员密码");
      return;
    }

    const result = await adminLogin({
      username: form.username.trim(),
      password: form.password,
    });

    if (result.success) {
      localStorage.setItem("isLogin", "true");
      localStorage.setItem("username", result.username || form.username.trim());
      localStorage.setItem("role", "admin");

      navigate("/admin");
    } else {
      alert(result.message || "管理员登录失败");
    }
  };

  return (
    <div className="home-container">
      <Navbar />

      <main className="auth-page">
        <form className="admin-login-box auth-box" onSubmit={handleAdminLogin}>
          <h2>管理员登录</h2>

          <input
            type="text"
            name="username"
            placeholder="管理员用户名"
            value={form.username}
            onChange={handleChange}
          />

          <input
            type="password"
            name="password"
            placeholder="管理员密码"
            value={form.password}
            onChange={handleChange}
          />

          <button type="submit">登录</button>

          <p className="form-tip">
            返回
            <Link className="auth-link" to="/login">
              用户登录
            </Link>
          </p>
        </form>
      </main>

      <Footer />
    </div>
  );
}