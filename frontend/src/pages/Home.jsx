import React from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

export default function Home() {
  return (
    <div className="home-container">
      <Navbar />
      <section className="hero-section">
        <h1>智慧空气质量预测平台</h1>
        <p>基于LSTM的城市PM2.5短期预测与预警系统</p>
        <div className="hero-buttons">
          <button onClick={() => window.location.href="/login"}>登录</button>
          <button onClick={() => window.location.href="/prediction"}>快速预测</button>
        </div>
      </section>

      <section className="features-section">
        <h2>系统功能概览</h2>
        <div className="feature-cards">
          <div className="card"><h3>实时监测</h3><p>查看各站点PM2.5浓度趋势</p></div>
          <div className="card"><h3>预测分析</h3><p>历史数据拟合 & 未来7天预测</p></div>
          <div className="card"><h3>预警提示</h3><p>根据空气质量等级生成自动预警</p></div>
        </div>
      </section>

      <Footer />
    </div>
  );
}