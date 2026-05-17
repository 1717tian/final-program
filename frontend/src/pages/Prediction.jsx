import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { fetchModels, fetchStations, fetchPrediction } from "../api/backend";

/* =========================
   日期工具
========================= */

function parseDateValue(dateValue) {
  if (!dateValue) return 0;

  const text = String(dateValue).trim().replaceAll("/", "-");
  const match = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);

  if (!match) {
    const time = new Date(text).getTime();
    return Number.isFinite(time) ? time : 0;
  }

  const year = Number(match[1]);
  const month = Number(match[2]) - 1;
  const day = Number(match[3]);

  return new Date(year, month, day).getTime();
}

function formatAxisDate(dateValue) {
  if (!dateValue) return "--";

  const text = String(dateValue).trim().replaceAll("/", "-");
  const match = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);

  if (!match) return text;

  const month = String(Number(match[2])).padStart(2, "0");
  const day = String(Number(match[3])).padStart(2, "0");

  return `${month}-${day}`;
}

function formatYearMonthDate(dateValue) {
  if (!dateValue) return "--";

  const text = String(dateValue).trim().replaceAll("/", "-");
  const match = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);

  if (!match) return text;

  const year = match[1];
  const month = String(Number(match[2])).padStart(2, "0");

  return `${year}-${month}`;
}

function sortByDate(list) {
  if (!Array.isArray(list)) return [];

  return [...list].sort((a, b) => {
    return parseDateValue(a.date) - parseDateValue(b.date);
  });
}

function sortLongFitByDate(list) {
  if (!Array.isArray(list)) return [];

  return [...list].sort((a, b) => {
    return parseDateValue(a.date) - parseDateValue(b.date);
  });
}

function getYearMonthLabelIndices(list, maxLabels = 8) {
  if (!Array.isArray(list) || list.length === 0) {
    return new Set();
  }

  const monthStartIndices = [];
  let previousMonth = "";

  list.forEach((item, index) => {
    const month = formatYearMonthDate(item.date);

    if (month !== previousMonth) {
      monthStartIndices.push(index);
      previousMonth = month;
    }
  });

  if (!monthStartIndices.includes(list.length - 1)) {
    monthStartIndices.push(list.length - 1);
  }

  if (monthStartIndices.length <= maxLabels) {
    return new Set(monthStartIndices);
  }

  const selected = [];

  for (let i = 0; i < maxLabels; i += 1) {
    const index = Math.round(
      (i * (monthStartIndices.length - 1)) / (maxLabels - 1)
    );

    selected.push(monthStartIndices[index]);
  }

  return new Set(selected);
}

/* =========================
   空气质量等级
========================= */

function getAirLevel(value) {
  if (value <= 35) return { text: "优", color: "#4ade80" };
  if (value <= 75) return { text: "良", color: "#22d3ee" };
  if (value <= 115) return { text: "轻度污染", color: "#facc15" };
  if (value <= 150) return { text: "中度污染", color: "#fb923c" };
  if (value <= 250) return { text: "重度污染", color: "#ef4444" };
  return { text: "严重污染", color: "#a855f7" };
}

function average(list) {
  if (!list.length) return null;

  const values = list
    .map((item) => Number(item.pm25))
    .filter((value) => Number.isFinite(value));

  if (!values.length) return null;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/* =========================
   SVG viewBox 交互容器
   用于长期拟合图
   不使用 CSS scale，避免放大模糊
========================= */

function InteractiveSvgFrame({ width, height, children }) {
  const viewportRef = useRef(null);

  const [viewBox, setViewBox] = useState({
    x: 0,
    y: 0,
    width,
    height,
  });

  const [dragging, setDragging] = useState(false);
  const [lastPoint, setLastPoint] = useState({ x: 0, y: 0 });

  const zoomPercent = Math.round((width / viewBox.width) * 100);

  const clampViewBox = (next) => {
    const minWidth = width / 10;
    const maxWidth = width;

    const minHeight = height / 10;
    const maxHeight = height;

    const newWidth = Math.min(maxWidth, Math.max(minWidth, next.width));
    const newHeight = Math.min(maxHeight, Math.max(minHeight, next.height));

    let newX = next.x;
    let newY = next.y;

    if (newX < 0) newX = 0;
    if (newY < 0) newY = 0;

    if (newX + newWidth > width) {
      newX = width - newWidth;
    }

    if (newY + newHeight > height) {
      newY = height - newHeight;
    }

    return {
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
    };
  };

  useEffect(() => {
    const el = viewportRef.current;

    if (!el) return undefined;

    const handleWheel = (event) => {
      event.preventDefault();
      event.stopPropagation();

      const rect = el.getBoundingClientRect();

      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      const svgX = viewBox.x + (mouseX / rect.width) * viewBox.width;
      const svgY = viewBox.y + (mouseY / rect.height) * viewBox.height;

      const zoomFactor = event.deltaY < 0 ? 0.85 : 1.18;

      const nextWidth = viewBox.width * zoomFactor;
      const nextHeight = viewBox.height * zoomFactor;

      const nextX = svgX - (mouseX / rect.width) * nextWidth;
      const nextY = svgY - (mouseY / rect.height) * nextHeight;

      setViewBox(
        clampViewBox({
          x: nextX,
          y: nextY,
          width: nextWidth,
          height: nextHeight,
        })
      );
    };

    el.addEventListener("wheel", handleWheel, {
      passive: false,
    });

    return () => {
      el.removeEventListener("wheel", handleWheel);
    };
  }, [viewBox]);

  const handleMouseDown = (event) => {
    event.preventDefault();

    setDragging(true);

    setLastPoint({
      x: event.clientX,
      y: event.clientY,
    });
  };

  const handleMouseMove = (event) => {
    if (!dragging || !viewportRef.current) return;

    event.preventDefault();

    const rect = viewportRef.current.getBoundingClientRect();

    const dx = event.clientX - lastPoint.x;
    const dy = event.clientY - lastPoint.y;

    const moveX = -(dx / rect.width) * viewBox.width;
    const moveY = -(dy / rect.height) * viewBox.height;

    setViewBox(
      clampViewBox({
        ...viewBox,
        x: viewBox.x + moveX,
        y: viewBox.y + moveY,
      })
    );

    setLastPoint({
      x: event.clientX,
      y: event.clientY,
    });
  };

  const handleMouseUp = () => {
    setDragging(false);
  };

  const handleZoomIn = () => {
    const nextWidth = viewBox.width * 0.8;
    const nextHeight = viewBox.height * 0.8;

    setViewBox(
      clampViewBox({
        x: viewBox.x + (viewBox.width - nextWidth) / 2,
        y: viewBox.y + (viewBox.height - nextHeight) / 2,
        width: nextWidth,
        height: nextHeight,
      })
    );
  };

  const handleZoomOut = () => {
    const nextWidth = viewBox.width * 1.25;
    const nextHeight = viewBox.height * 1.25;

    setViewBox(
      clampViewBox({
        x: viewBox.x - (nextWidth - viewBox.width) / 2,
        y: viewBox.y - (nextHeight - viewBox.height) / 2,
        width: nextWidth,
        height: nextHeight,
      })
    );
  };

  const handleReset = () => {
    setViewBox({
      x: 0,
      y: 0,
      width,
      height,
    });
  };

  return (
    <div style={styles.interactiveBox}>
      <div style={styles.chartToolbar}>
        <button style={styles.chartToolButton} onClick={handleZoomIn}>
          放大
        </button>

        <button style={styles.chartToolButton} onClick={handleZoomOut}>
          缩小
        </button>

        <button style={styles.chartToolButton} onClick={handleReset}>
          重置
        </button>

        <span style={styles.zoomText}>缩放：{zoomPercent}%</span>
      </div>

      <div
        ref={viewportRef}
        style={{
          ...styles.interactiveViewport,
          cursor: dragging ? "grabbing" : "grab",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {children(viewBox)}
      </div>
    </div>
  );
}

/* =========================
   图1：历史真实值 + 未来预测趋势
   静态图，不缩放、不拖动
========================= */

function SimpleChart({ history, forecast }) {
  const sortedHistory = sortByDate(history).filter((item) =>
    Number.isFinite(Number(item.pm25))
  );

  const sortedForecast = sortByDate(forecast).filter((item) =>
    Number.isFinite(Number(item.pm25))
  );

  const all = [
    ...sortedHistory.map((item) => ({ ...item, type: "history" })),
    ...sortedForecast.map((item) => ({ ...item, type: "forecast" })),
  ];

  if (!all.length) {
    return <div style={styles.emptyChart}>请选择站点和模型后点击“开始预测”</div>;
  }

  const width = 1000;
  const height = 380;
  const padding = 55;

  const values = all
    .map((item) => Number(item.pm25))
    .filter((value) => Number.isFinite(value));

  const min = Math.max(0, Math.min(...values) - 15);
  const max = Math.max(...values) + 15;
  const range = max - min || 1;

  const getX = (index) => {
    if (all.length === 1) return width / 2;

    return padding + (index / (all.length - 1)) * (width - padding * 2);
  };

  const getY = (value) => {
    return height - padding - ((value - min) / range) * (height - padding * 2);
  };

  const historyPoints = sortedHistory
    .map((item, index) => `${getX(index)},${getY(Number(item.pm25))}`)
    .join(" ");

  const forecastStartIndex = Math.max(sortedHistory.length - 1, 0);

  const forecastLineData =
    sortedHistory.length > 0 && sortedForecast.length > 0
      ? [sortedHistory[sortedHistory.length - 1], ...sortedForecast]
      : sortedForecast;

  const forecastPoints = forecastLineData
    .map((item, index) => {
      return `${getX(forecastStartIndex + index)},${getY(Number(item.pm25))}`;
    })
    .join(" ");

  return (
    <div style={styles.chartScroll}>
      <svg viewBox={`0 0 ${width} ${height}`} style={styles.svg}>
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.45)"
          strokeWidth="2"
        />

        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.45)"
          strokeWidth="2"
        />

        {[0, 1, 2, 3, 4].map((tick) => {
          const y = padding + tick * ((height - padding * 2) / 4);
          const value = Math.round(max - tick * (range / 4));

          return (
            <g key={tick}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="rgba(255,255,255,0.12)"
              />

              <text
                x={padding - 12}
                y={y + 5}
                fill="rgba(255,255,255,0.75)"
                fontSize="14"
                textAnchor="end"
              >
                {value}
              </text>
            </g>
          );
        })}

        {sortedHistory.length > 0 && sortedForecast.length > 0 && (
          <line
            x1={getX(sortedHistory.length - 1)}
            y1={padding}
            x2={getX(sortedHistory.length - 1)}
            y2={height - padding}
            stroke="rgba(255,255,255,0.45)"
            strokeDasharray="8 8"
          />
        )}

        {historyPoints && (
          <polyline
            points={historyPoints}
            fill="none"
            stroke="#00ffff"
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {forecastPoints && (
          <polyline
            points={forecastPoints}
            fill="none"
            stroke="#facc15"
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="10 6"
          />
        )}

        {all.map((item, index) => (
          <circle
            key={`${item.date}-${index}`}
            cx={getX(index)}
            cy={getY(Number(item.pm25))}
            r="5"
            fill={item.type === "history" ? "#00ffff" : "#facc15"}
          />
        ))}

        {all.map((item, index) => {
          if (index !== 0 && index !== all.length - 1 && index % 3 !== 0) {
            return null;
          }

          return (
            <text
              key={`date-${index}`}
              x={getX(index)}
              y={height - 18}
              fill="rgba(255,255,255,0.72)"
              fontSize="13"
              textAnchor="middle"
            >
              {formatAxisDate(item.date)}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

/* =========================
   图2：长期历史拟合效果
   SVG viewBox 矢量缩放
========================= */

function LongFitChart({ longFit }) {
  if (!longFit || longFit.length === 0) {
    return <div style={styles.emptyChart}>暂无长期拟合数据，请先点击“开始预测”</div>;
  }

  const sortedLongFit = sortLongFitByDate(longFit).filter((item) => {
    return (
      Number.isFinite(Number(item.actual_pm25)) &&
      Number.isFinite(Number(item.predicted_pm25))
    );
  });

  if (!sortedLongFit.length) {
    return <div style={styles.emptyChart}>长期拟合数据格式异常</div>;
  }

  const width = 1100;
  const height = 420;
  const padding = 65;

  const values = sortedLongFit.flatMap((item) => [
    Number(item.actual_pm25),
    Number(item.predicted_pm25),
  ]);

  const min = Math.max(0, Math.min(...values) - 15);
  const max = Math.max(...values) + 15;
  const range = max - min || 1;

  const getX = (index) => {
    if (sortedLongFit.length === 1) return width / 2;

    return padding + (index / (sortedLongFit.length - 1)) * (width - padding * 2);
  };

  const getY = (value) => {
    return height - padding - ((value - min) / range) * (height - padding * 2);
  };

  const actualPoints = sortedLongFit
    .map((item, index) => {
      return `${getX(index)},${getY(Number(item.actual_pm25))}`;
    })
    .join(" ");

  const predictedPoints = sortedLongFit
    .map((item, index) => {
      return `${getX(index)},${getY(Number(item.predicted_pm25))}`;
    })
    .join(" ");

  const labelIndices = getYearMonthLabelIndices(sortedLongFit, 8);

  return (
    <InteractiveSvgFrame width={width} height={height}>
      {(viewBox) => (
        <svg
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          style={styles.longSvg}
          preserveAspectRatio="xMidYMid meet"
        >
          <line
            x1={padding}
            y1={height - padding}
            x2={width - padding}
            y2={height - padding}
            stroke="rgba(255,255,255,0.45)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />

          <line
            x1={padding}
            y1={padding}
            x2={padding}
            y2={height - padding}
            stroke="rgba(255,255,255,0.45)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />

          {[0, 1, 2, 3, 4].map((tick) => {
            const y = padding + tick * ((height - padding * 2) / 4);
            const value = Math.round(max - tick * (range / 4));

            return (
              <g key={tick}>
                <line
                  x1={padding}
                  y1={y}
                  x2={width - padding}
                  y2={y}
                  stroke="rgba(255,255,255,0.12)"
                  vectorEffect="non-scaling-stroke"
                />

                <text
                  x={padding - 12}
                  y={y + 5}
                  fill="rgba(255,255,255,0.75)"
                  fontSize="14"
                  textAnchor="end"
                >
                  {value}
                </text>
              </g>
            );
          })}

          <polyline
            points={actualPoints}
            fill="none"
            stroke="#4ade80"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />

          <polyline
            points={predictedPoints}
            fill="none"
            stroke="#facc15"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="9 6"
            vectorEffect="non-scaling-stroke"
          />

          {sortedLongFit.map((item, index) => {
            if (!labelIndices.has(index)) return null;

            return (
              <text
                key={`long-fit-date-${index}`}
                x={getX(index)}
                y={height - 22}
                fill="rgba(255,255,255,0.72)"
                fontSize="13"
                textAnchor="middle"
              >
                {formatYearMonthDate(item.date)}
              </text>
            );
          })}
        </svg>
      )}
    </InteractiveSvgFrame>
  );
}

/* =========================
   页面主体
========================= */

export default function Prediction() {
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("featured_direct7");

  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState("");

  const [dataVersion, setDataVersion] = useState("");
  const [lastDate, setLastDate] = useState("");

  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [longFit, setLongFit] = useState([]);

  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    const isLogin = localStorage.getItem("isLogin");

    if (isLogin !== "true") {
      alert("请先登录后再访问预测页面");
      navigate("/login");
      return;
    }

    loadInitialData();
  }, [navigate]);

  const loadInitialData = async () => {
    const modelResult = await fetchModels();

    if (modelResult.success && Array.isArray(modelResult.models)) {
      setModels(modelResult.models);

      const defaultAvailable =
        modelResult.models.find(
          (item) => item.key === modelResult.default_model && item.available
        ) ||
        modelResult.models.find((item) => item.available) ||
        modelResult.models[0];

      if (defaultAvailable) {
        setSelectedModel(defaultAvailable.key);
      }
    } else {
      setMessage(modelResult.message || "模型列表加载失败");
    }

    const stationResult = await fetchStations();

    if (
      stationResult.success &&
      Array.isArray(stationResult.stations) &&
      stationResult.stations.length > 0
    ) {
      const sortedStations = [...stationResult.stations].sort((a, b) => {
        return String(a.name || "").localeCompare(String(b.name || ""));
      });

      setStations(sortedStations);

      const first = sortedStations[0];

      setSelectedStation(first.id);
      setDataVersion(first.version || "");
      setLastDate(first.last_date || "");
    } else {
      setMessage(stationResult.message || "未获取到站点CSV数据");
    }
  };

  const handleStationChange = (event) => {
    const stationId = event.target.value;

    setSelectedStation(stationId);

    const station = stations.find((item) => item.id === stationId);

    if (station) {
      setDataVersion(station.version || "");
      setLastDate(station.last_date || "");
    }

    setHistory([]);
    setForecast([]);
    setLongFit([]);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);

    setHistory([]);
    setForecast([]);
    setLongFit([]);
  };

  const handlePredict = async () => {
    if (!selectedStation) {
      alert("请先选择监测站点");
      return;
    }

    if (!selectedModel) {
      alert("请先选择预测模型");
      return;
    }

    setLoadingPrediction(true);
    setMessage("");

    const result = await fetchPrediction({
      station: selectedStation,
      model_key: selectedModel,
      history_days: 14,
      days: 7,
    });

    if (result.success) {
      const sortedHistory = sortByDate(result.history || []);
      const sortedForecast = sortByDate(result.forecast || []);
      const sortedLongFit = sortLongFitByDate(result.long_fit || []);

      setHistory(sortedHistory);
      setForecast(sortedForecast);
      setLongFit(sortedLongFit);

      setDataVersion(result.version || "");
      setLastDate(result.last_date || "");
    } else {
      setMessage(result.message || "预测失败");

      setHistory([]);
      setForecast([]);
      setLongFit([]);
    }

    setLoadingPrediction(false);
  };

  const currentStation = stations.find((item) => item.id === selectedStation);
  const currentModel = models.find((item) => item.key === selectedModel);

  const stats = useMemo(() => {
    const sortedHistory = sortByDate(history);
    const sortedForecast = sortByDate(forecast);

    const lastHistory = sortedHistory.length
      ? Number(sortedHistory[sortedHistory.length - 1].pm25)
      : null;

    const avgForecast = average(sortedForecast);

    const maxForecast = sortedForecast.length
      ? Math.max(...sortedForecast.map((item) => Number(item.pm25)))
      : null;

    return {
      lastHistory,
      avgForecast,
      maxForecast,
      level:
        avgForecast !== null
          ? getAirLevel(avgForecast)
          : { text: "--", color: "#ffffff" },
    };
  }, [history, forecast]);

  return (
    <div className="home-container">
      <Navbar />

      <main style={styles.page}>
        <section style={styles.header}>
          <div>
            <h1 style={styles.title}>PM2.5 浓度预测分析</h1>
            <p style={styles.subtitle}>
              支持切换监测站点与预测模型，展示未来7天预测和长期历史拟合效果。
            </p>
          </div>

          <button
            style={styles.logoutButton}
            onClick={() => {
              localStorage.removeItem("isLogin");
              localStorage.removeItem("username");
              localStorage.removeItem("role");
              navigate("/login");
            }}
          >
            退出登录
          </button>
        </section>

        <section style={styles.controlPanel}>
          <div style={styles.controlItem}>
            <label style={styles.label}>监测站点</label>
            <select
              style={styles.select}
              value={selectedStation}
              onChange={handleStationChange}
            >
              {stations.map((station) => (
                <option key={station.id} value={station.id}>
                  {station.name}
                </option>
              ))}
            </select>
          </div>

          <div style={styles.controlItem}>
            <label style={styles.label}>预测模型</label>
            <select
              style={styles.select}
              value={selectedModel}
              onChange={handleModelChange}
            >
              {models.map((model) => (
                <option
                  key={model.key}
                  value={model.key}
                  disabled={!model.available}
                >
                  {model.label}
                  {model.available ? "" : "（文件缺失）"}
                </option>
              ))}
            </select>
          </div>

          <div style={styles.versionBox}>
            <span style={styles.versionLabel}>当前站点文件</span>
            <strong>{currentStation?.file || "--"}</strong>
          </div>

          <div style={styles.versionBox}>
            <span style={styles.versionLabel}>最后数据日期</span>
            <strong>{lastDate || "--"}</strong>
          </div>

          <div style={styles.versionBox}>
            <span style={styles.versionLabel}>数据版本</span>
            <strong>{dataVersion || "--"}</strong>
          </div>

          <button
            style={styles.primaryButton}
            onClick={handlePredict}
            disabled={loadingPrediction}
          >
            {loadingPrediction ? "预测中..." : "开始预测"}
          </button>
        </section>

        {currentModel && (
          <section style={styles.modelInfoPanel}>
            <strong style={{ color: "#00ffff" }}>当前模型：</strong>

            <span>{currentModel.label}</span>

            <span style={styles.modelInfoText}>
              模型文件：{currentModel.model_file}
            </span>

            <span style={styles.modelInfoText}>
              归一化器：{currentModel.scaler_file}
            </span>
          </section>
        )}

        {message && <div style={styles.message}>{message}</div>}

        <section style={styles.statGrid}>
          <div style={styles.statCard}>
            <span style={styles.statLabel}>最近历史值</span>
            <strong style={styles.statValue}>
              {stats.lastHistory !== null ? stats.lastHistory.toFixed(1) : "--"}
            </strong>
            <span style={styles.unit}>μg/m³</span>
          </div>

          <div style={styles.statCard}>
            <span style={styles.statLabel}>未来7天均值</span>
            <strong style={styles.statValue}>
              {stats.avgForecast !== null ? stats.avgForecast.toFixed(1) : "--"}
            </strong>
            <span style={styles.unit}>μg/m³</span>
          </div>

          <div style={styles.statCard}>
            <span style={styles.statLabel}>预测峰值</span>
            <strong style={styles.statValue}>
              {stats.maxForecast !== null ? stats.maxForecast.toFixed(1) : "--"}
            </strong>
            <span style={styles.unit}>μg/m³</span>
          </div>

          <div style={styles.statCard}>
            <span style={styles.statLabel}>空气质量等级</span>
            <strong style={{ ...styles.statValue, color: stats.level.color }}>
              {stats.level.text}
            </strong>
            <span style={styles.unit}>综合判断</span>
          </div>
        </section>

        <section style={styles.chartPanel}>
          <div style={styles.chartHeader}>
            <h2 style={styles.chartTitle}>历史真实值与未来预测趋势</h2>

            <div style={styles.legend}>
              <span style={styles.legendHistory}></span>历史真实值
              <span style={styles.legendForecast}></span>未来预测值
            </div>
          </div>

          <SimpleChart history={history} forecast={forecast} />
        </section>

        <section style={styles.tablePanel}>
          <h2 style={styles.chartTitle}>未来7天预测明细</h2>

          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>日期</th>
                  <th style={styles.th}>预测PM2.5</th>
                  <th style={styles.th}>空气等级</th>
                  <th style={styles.th}>预警建议</th>
                </tr>
              </thead>

              <tbody>
                {forecast.length === 0 ? (
                  <tr>
                    <td style={styles.td} colSpan="4">
                      暂无预测明细
                    </td>
                  </tr>
                ) : (
                  sortByDate(forecast).map((item) => {
                    const level = getAirLevel(Number(item.pm25));

                    return (
                      <tr key={item.date}>
                        <td style={styles.td}>{item.date}</td>

                        <td style={styles.td}>
                          {Number(item.pm25).toFixed(1)} μg/m³
                        </td>

                        <td style={{ ...styles.td, color: level.color }}>
                          {level.text}
                        </td>

                        <td style={styles.td}>
                          {Number(item.pm25) > 75
                            ? "建议减少户外活动，关注空气质量变化"
                            : "空气质量较好，可正常活动"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section style={styles.longFitPanel}>
          <div style={styles.chartHeader}>
            <h2 style={styles.chartTitle}>长期历史拟合效果</h2>

            <div style={styles.legend}>
              <span style={styles.legendActual}></span>历史真实值
              <span style={styles.legendFit}></span>模型历史预测值
            </div>
          </div>

          <p style={styles.longFitDesc}>
            该图展示当前所选模型在历史时间序列上的拟合效果，用于观察模型预测曲线与真实
            PM2.5 浓度变化趋势的一致性。长期图使用 SVG 矢量缩放，放大后不会模糊；可用鼠标滚轮缩放，按住鼠标拖动查看局部细节。
          </p>

          <LongFitChart longFit={longFit} />
        </section>
      </main>

      <Footer />
    </div>
  );
}

/* =========================
   样式
========================= */

const styles = {
  page: {
    flex: 1,
    width: "100%",
    padding: "110px 40px 50px",
    color: "#ffffff",
  },

  header: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "20px",
  },

  title: {
    margin: 0,
    color: "#00ffff",
    fontSize: "34px",
    textShadow: "0 0 10px rgba(0,255,255,0.65)",
  },

  subtitle: {
    marginTop: "10px",
    color: "rgba(255,255,255,0.82)",
  },

  logoutButton: {
    padding: "10px 18px",
    border: "1px solid rgba(0,255,255,0.55)",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.45)",
    color: "#00ffff",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  controlPanel: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    padding: "22px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.58)",
    display: "flex",
    alignItems: "end",
    gap: "18px",
    flexWrap: "wrap",
  },

  controlItem: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    minWidth: "240px",
  },

  label: {
    color: "#00ffff",
    fontWeight: "bold",
  },

  select: {
    height: "42px",
    padding: "0 12px",
    borderRadius: "6px",
    border: "none",
    outline: "none",
    minWidth: "240px",
  },

  versionBox: {
    minWidth: "170px",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    color: "#ffffff",
  },

  versionLabel: {
    color: "#00ffff",
    fontSize: "13px",
  },

  primaryButton: {
    height: "42px",
    padding: "0 26px",
    border: "none",
    borderRadius: "6px",
    color: "#ffffff",
    cursor: "pointer",
    background: "linear-gradient(90deg, #00bfff, #007fff)",
  },

  modelInfoPanel: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    padding: "14px 20px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.55)",
    display: "flex",
    flexWrap: "wrap",
    gap: "16px",
    color: "rgba(255,255,255,0.85)",
  },

  modelInfoText: {
    color: "rgba(255,255,255,0.78)",
  },

  message: {
    maxWidth: "1200px",
    margin: "0 auto 20px",
    padding: "12px 18px",
    borderRadius: "8px",
    color: "#facc15",
    background: "rgba(0,0,0,0.55)",
  },

  statGrid: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
    gap: "18px",
  },

  statCard: {
    padding: "22px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.58)",
    boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
  },

  statLabel: {
    display: "block",
    color: "rgba(255,255,255,0.78)",
    marginBottom: "8px",
  },

  statValue: {
    display: "inline-block",
    fontSize: "30px",
    color: "#00ffff",
    marginRight: "8px",
  },

  unit: {
    color: "rgba(255,255,255,0.72)",
  },

  chartPanel: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    padding: "24px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.6)",
  },

  tablePanel: {
    maxWidth: "1200px",
    margin: "0 auto 24px",
    padding: "24px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.6)",
  },

  longFitPanel: {
    maxWidth: "1200px",
    margin: "0 auto 0",
    padding: "24px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.6)",
  },

  longFitDesc: {
    marginTop: "8px",
    marginBottom: "16px",
    color: "rgba(255,255,255,0.75)",
    fontSize: "14px",
    lineHeight: 1.6,
  },

  chartHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap",
    marginBottom: "10px",
  },

  chartTitle: {
    margin: 0,
    color: "#ffffff",
    fontSize: "22px",
  },

  legend: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "rgba(255,255,255,0.8)",
    fontSize: "14px",
  },

  legendHistory: {
    width: "24px",
    height: "4px",
    background: "#00ffff",
    borderRadius: "10px",
    display: "inline-block",
  },

  legendForecast: {
    width: "24px",
    height: "4px",
    background: "#facc15",
    borderRadius: "10px",
    display: "inline-block",
    marginLeft: "12px",
  },

  legendActual: {
    width: "24px",
    height: "4px",
    background: "#4ade80",
    borderRadius: "10px",
    display: "inline-block",
  },

  legendFit: {
    width: "24px",
    height: "4px",
    background: "#facc15",
    borderRadius: "10px",
    display: "inline-block",
    marginLeft: "12px",
  },

  chartScroll: {
    width: "100%",
    overflowX: "auto",
  },

  svg: {
    width: "100%",
    minWidth: "720px",
    height: "380px",
    display: "block",
  },

  longSvg: {
    width: "100%",
    height: "420px",
    display: "block",
  },

  interactiveBox: {
    width: "100%",
    borderRadius: "10px",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.22)",
  },

  chartToolbar: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "10px",
    borderBottom: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.35)",
  },

  chartToolButton: {
    padding: "6px 12px",
    border: "1px solid rgba(0,255,255,0.45)",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.4)",
    color: "#00ffff",
    cursor: "pointer",
  },

  zoomText: {
    marginLeft: "8px",
    color: "rgba(255,255,255,0.72)",
    fontSize: "13px",
  },

  interactiveViewport: {
    width: "100%",
    height: "470px",
    overflow: "hidden",
    position: "relative",
    userSelect: "none",
    overscrollBehavior: "contain",
    touchAction: "none",
  },

  emptyChart: {
    height: "300px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    border: "1px dashed rgba(255,255,255,0.25)",
    borderRadius: "10px",
    color: "rgba(255,255,255,0.75)",
  },

  tableWrap: {
    marginTop: "16px",
    overflowX: "auto",
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    minWidth: "760px",
  },

  th: {
    padding: "14px",
    textAlign: "left",
    color: "#00ffff",
    borderBottom: "1px solid rgba(255,255,255,0.25)",
  },

  td: {
    padding: "14px",
    borderBottom: "1px solid rgba(255,255,255,0.12)",
    color: "rgba(255,255,255,0.86)",
  },
};