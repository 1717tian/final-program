import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { register } from "../api/backend";

export default function Register() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: "",
    password: "",
    confirmPassword: "",
  });

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleRegister = async (e) => {
    e.preventDefault();

    if (!form.username.trim()) {
      alert("请输入用户名");
      return;
    }

    if (!form.password.trim()) {
      alert("请输入密码");
      return;
    }

    if (!form.confirmPassword.trim()) {
      alert("请确认密码");
      return;
    }

    if (form.password !== form.confirmPassword) {
      alert("两次输入的密码不一致");
      return;
    }

    const result = await register({
      username: form.username.trim(),
      password: form.password,
    });

    if (result.success) {
      alert("注册成功，请登录");
      navigate("/login");
    } else {
      alert(result.message || "注册失败");
    }
  };

  return (
    <div className="home-container">
      <Navbar />

      <main className="auth-page">
        <form className="register-box auth-box" onSubmit={handleRegister}>
          <h2>用户注册</h2>

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

          <input
            type="password"
            name="confirmPassword"
            placeholder="确认密码"
            value={form.confirmPassword}
            onChange={handleChange}
          />

          <button type="submit">注册</button>

          <p className="form-tip">
            已有账号？
            <Link className="auth-link" to="/login">
              返回登录
            </Link>
          </p>
        </form>
      </main>

      <Footer />
    </div>
  );
}