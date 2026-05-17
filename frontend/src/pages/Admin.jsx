import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import {
  fetchDatasets,
  uploadDataset,
  createDataset,
  renameDataset,
  deleteDataset,
  previewDataset,
  saveDatasetContent,
  fetchAdminModels,
  updateModelEnabled,
  startTraining,
  fetchTrainJobs,
  fetchTrainJobLog,
  terminateTrainJob,
  deleteTrainJob,
  renameModel,
  deleteModel,
} from "../api/backend";

const MODEL_OPTIONS = [
  {
    key: "old_direct7",
    label: "普通多变量LSTM",
  },
  {
    key: "featured_direct7",
    label: "特征增强LSTM",
  },
  {
    key: "featured_warning_direct7",
    label: "预警导向特征增强LSTM",
  },
];

const DEFAULT_CSV_COLUMNS = ["date", "pm25", "pm10", "o3", "no2", "so2", "co"];

export default function Admin() {
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState("datasets");

  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);

  const [selectedFile, setSelectedFile] = useState("");
  const [preview, setPreview] = useState({
    columns: [],
    rows: [],
  });

  const [csvEditMode, setCsvEditMode] = useState(false);
  const [editColumns, setEditColumns] = useState([]);
  const [editRows, setEditRows] = useState([]);
  const [savingCsv, setSavingCsv] = useState(false);

  const [newDatasetName, setNewDatasetName] = useState("");
  const [uploading, setUploading] = useState(false);

  const [selectedJobId, setSelectedJobId] = useState("");
  const [trainLog, setTrainLog] = useState("");
  const [trainLogLoading, setTrainLogLoading] = useState(false);

  const logBoxRef = useRef(null);
  const [autoScrollLog, setAutoScrollLog] = useState(true);

  const [openModelMenuKey, setOpenModelMenuKey] = useState("");

  const [trainForm, setTrainForm] = useState({
    model_name: "pm25_lstm_new",
    model_key: "featured_warning_direct7",
    dataset_filenames: [],
    epochs: 150,
    batch_size: 32,
    learning_rate: 0.0005,
    hidden_size: 128,
    num_layers: 2,
    dropout: 0.2,
    seq_length: 21,
    pred_days: 7,
    remark: "",
  });

  const selectedDatasetInfo = useMemo(() => {
    return datasets.find((item) => item.filename === selectedFile);
  }, [datasets, selectedFile]);

  const selectedJob = useMemo(() => {
    return jobs.find((item) => item.job_id === selectedJobId);
  }, [jobs, selectedJobId]);

  useEffect(() => {
    const isLogin = localStorage.getItem("isLogin");
    const role = localStorage.getItem("role");

    if (isLogin !== "true" || role !== "admin") {
      alert("请先使用管理员账号登录");
      navigate("/admin-login");
      return;
    }

    loadAll();
  }, [navigate]);

  useEffect(() => {
    if (activeTab !== "training" || !selectedJobId) return undefined;

    loadTrainLog(selectedJobId);

    const timer = setInterval(() => {
      loadJobs();
      loadTrainLog(selectedJobId);
    }, 2500);

    return () => clearInterval(timer);
  }, [activeTab, selectedJobId]);

  useEffect(() => {
    if (!autoScrollLog) return;

    const logBox = logBoxRef.current;

    if (!logBox) return;

    requestAnimationFrame(() => {
      logBox.scrollTop = logBox.scrollHeight;
    });
  }, [trainLog, selectedJobId, autoScrollLog]);

  const loadAll = async () => {
    await Promise.all([loadDatasets(), loadModels(), loadJobs()]);
  };

  const loadDatasets = async () => {
    const result = await fetchDatasets();

    if (result.success) {
      const list = result.datasets || [];
      setDatasets(list);

      if (trainForm.dataset_filenames.length === 0 && list.length > 0) {
        setTrainForm((prev) => ({
          ...prev,
          dataset_filenames: [list[0].filename],
        }));
      }
    } else {
      alert(result.message);
    }
  };

  const loadModels = async () => {
    const result = await fetchAdminModels();

    if (result.success) {
      setModels(result.models || []);
    } else {
      alert(result.message);
    }
  };

  const loadJobs = async () => {
    const result = await fetchTrainJobs();

    if (result.success) {
      setJobs(result.jobs || []);
    }
  };

  const loadTrainLog = async (jobId) => {
    if (!jobId) return;

    setTrainLogLoading(true);

    const result = await fetchTrainJobLog(jobId, 500);

    setTrainLogLoading(false);

    if (result.success) {
      setTrainLog(result.log || "");
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);

    const result = await uploadDataset(formData);

    setUploading(false);
    event.target.value = "";

    if (result.success) {
      alert("CSV导入成功");
      await loadDatasets();

      if (result.dataset?.filename) {
        await handlePreviewDataset(result.dataset.filename);
      }
    } else {
      alert(result.message);
    }
  };

  const handleCreateDataset = async () => {
    if (!newDatasetName.trim()) {
      alert("请输入新建CSV文件名");
      return;
    }

    const result = await createDataset({
      filename: newDatasetName.trim(),
      columns: DEFAULT_CSV_COLUMNS,
    });

    if (result.success) {
      alert("新建成功");
      setNewDatasetName("");
      await loadDatasets();

      if (result.dataset?.filename) {
        await handlePreviewDataset(result.dataset.filename);
        setCsvEditMode(true);
        setEditColumns(DEFAULT_CSV_COLUMNS);
        setEditRows([]);
      }
    } else {
      alert(result.message);
    }
  };

  const handleRenameDataset = async (filename) => {
    const newName = window.prompt("请输入新的CSV文件名", filename);

    if (!newName) return;

    const result = await renameDataset(filename, newName);

    if (result.success) {
      alert("重命名成功");

      if (selectedFile === filename) {
        setSelectedFile(result.dataset.filename);
      }

      setTrainForm((prev) => ({
        ...prev,
        dataset_filenames: prev.dataset_filenames.map((item) =>
          item === filename ? result.dataset.filename : item
        ),
      }));

      await loadDatasets();
    } else {
      alert(result.message);
    }
  };

  const handleDeleteDataset = async (filename) => {
    const ok = window.confirm(`确认删除数据集：${filename}？`);

    if (!ok) return;

    const result = await deleteDataset(filename);

    if (result.success) {
      alert("删除成功");

      if (selectedFile === filename) {
        setSelectedFile("");
        setPreview({
          columns: [],
          rows: [],
        });
        setCsvEditMode(false);
        setEditColumns([]);
        setEditRows([]);
      }

      setTrainForm((prev) => ({
        ...prev,
        dataset_filenames: prev.dataset_filenames.filter(
          (item) => item !== filename
        ),
      }));

      await loadDatasets();
    } else {
      alert(result.message);
    }
  };

  const handlePreviewDataset = async (filename) => {
    setSelectedFile(filename);
    setCsvEditMode(false);

    const result = await previewDataset(filename, 500);

    if (result.success) {
      const columns = result.columns || [];
      const rows = result.rows || [];

      setPreview({
        columns,
        rows,
      });

      setEditColumns(columns);
      setEditRows(
        rows.map((row) =>
          columns.map((col) => {
            const value = row[col];
            return value === null || value === undefined ? "" : String(value);
          })
        )
      );
    } else {
      alert(result.message);
    }
  };

  const enterCsvEditMode = () => {
    if (!selectedFile) {
      alert("请先选择一个CSV文件");
      return;
    }

    let columns = preview.columns || [];
    let rows = preview.rows || [];

    if (columns.length === 0) {
      columns = DEFAULT_CSV_COLUMNS;
      rows = [];
    }

    setEditColumns(columns);
    setEditRows(
      rows.map((row) =>
        columns.map((col) => {
          const value = row[col];
          return value === null || value === undefined ? "" : String(value);
        })
      )
    );

    setCsvEditMode(true);
  };

  const cancelCsvEditMode = () => {
    setCsvEditMode(false);

    const columns = preview.columns || [];
    const rows = preview.rows || [];

    setEditColumns(columns);
    setEditRows(
      rows.map((row) =>
        columns.map((col) => {
          const value = row[col];
          return value === null || value === undefined ? "" : String(value);
        })
      )
    );
  };

  const changeEditCell = (rowIndex, colIndex, value) => {
    setEditRows((prev) =>
      prev.map((row, rIndex) =>
        rIndex === rowIndex
          ? row.map((cell, cIndex) => (cIndex === colIndex ? value : cell))
          : row
      )
    );
  };

  const changeEditColumnName = (colIndex, value) => {
    setEditColumns((prevColumns) => {
      const oldName = prevColumns[colIndex];
      const newName = value;

      const nextColumns = prevColumns.map((col, index) =>
        index === colIndex ? newName : col
      );

      setEditRows((prevRows) =>
        prevRows.map((row) =>
          row.map((cell, index) => {
            if (index === colIndex) {
              return cell;
            }

            return cell;
          })
        )
      );

      return nextColumns;
    });
  };

  const addCsvRow = () => {
    const columns = editColumns.length > 0 ? editColumns : DEFAULT_CSV_COLUMNS;

    if (editColumns.length === 0) {
      setEditColumns(columns);
    }

    setEditRows((prev) => [...prev, columns.map(() => "")]);
  };

  const deleteCsvRow = (rowIndex) => {
    const ok = window.confirm(`确认删除第 ${rowIndex + 1} 行？`);

    if (!ok) return;

    setEditRows((prev) => prev.filter((_, index) => index !== rowIndex));
  };

  const addCsvColumn = () => {
    const name = window.prompt("请输入新增列名", `column_${editColumns.length + 1}`);

    if (!name) return;

    const newName = name.trim();

    if (!newName) {
      alert("列名不能为空");
      return;
    }

    if (editColumns.includes(newName)) {
      alert("列名已存在");
      return;
    }

    setEditColumns((prev) => [...prev, newName]);
    setEditRows((prev) => prev.map((row) => [...row, ""]));
  };

  const deleteCsvColumn = (colIndex) => {
    if (editColumns.length <= 1) {
      alert("CSV至少保留一列");
      return;
    }

    const colName = editColumns[colIndex];
    const ok = window.confirm(`确认删除列：${colName}？`);

    if (!ok) return;

    setEditColumns((prev) => prev.filter((_, index) => index !== colIndex));
    setEditRows((prev) =>
      prev.map((row) => row.filter((_, index) => index !== colIndex))
    );
  };

  const saveCsvEditContent = async () => {
    if (!selectedFile) {
      alert("请先选择CSV文件");
      return;
    }

    const columns = editColumns.map((col) => String(col).trim());

    if (columns.length === 0) {
      alert("CSV至少需要一列");
      return;
    }

    if (columns.some((col) => !col)) {
      alert("列名不能为空");
      return;
    }

    const uniqueColumns = new Set(columns);

    if (uniqueColumns.size !== columns.length) {
      alert("列名不能重复");
      return;
    }

    const rows = editRows.map((row) => {
      const item = {};

      columns.forEach((col, index) => {
        item[col] = row[index] ?? "";
      });

      return item;
    });

    setSavingCsv(true);

    const result = await saveDatasetContent(selectedFile, {
      columns,
      rows,
    });

    setSavingCsv(false);

    if (result.success) {
      alert("CSV保存成功");
      setCsvEditMode(false);
      await loadDatasets();
      await handlePreviewDataset(selectedFile);
    } else {
      alert(result.message || "CSV保存失败");
    }
  };

  const handleTrainFormChange = (field, value) => {
    setTrainForm((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const toggleTrainingDataset = (filename) => {
    setTrainForm((prev) => {
      const exists = prev.dataset_filenames.includes(filename);

      return {
        ...prev,
        dataset_filenames: exists
          ? prev.dataset_filenames.filter((item) => item !== filename)
          : [...prev.dataset_filenames, filename],
      };
    });
  };

  const selectAllDatasets = () => {
    setTrainForm((prev) => ({
      ...prev,
      dataset_filenames: datasets.map((item) => item.filename),
    }));
  };

  const clearSelectedDatasets = () => {
    setTrainForm((prev) => ({
      ...prev,
      dataset_filenames: [],
    }));
  };

  const handleStartTraining = async () => {
    if (!trainForm.model_name.trim()) {
      alert("请输入模型名称");
      return;
    }

    if (!trainForm.dataset_filenames || trainForm.dataset_filenames.length === 0) {
      alert("请至少选择一个训练数据集");
      return;
    }

    const payload = {
      ...trainForm,
      dataset_filenames: trainForm.dataset_filenames,
      epochs: Number(trainForm.epochs),
      batch_size: Number(trainForm.batch_size),
      learning_rate: Number(trainForm.learning_rate),
      hidden_size: Number(trainForm.hidden_size),
      num_layers: Number(trainForm.num_layers),
      dropout: Number(trainForm.dropout),
      seq_length: Number(trainForm.seq_length),
      pred_days: Number(trainForm.pred_days),
    };

    const result = await startTraining(payload);

    if (result.success) {
      alert("训练任务已提交");
      await loadJobs();

      if (result.job?.job_id) {
        setSelectedJobId(result.job.job_id);
        await loadTrainLog(result.job.job_id);
      }

      setActiveTab("training");
    } else {
      alert(result.message);
    }
  };

  const handleTerminateSelectedJob = async () => {
    if (!selectedJobId) {
      alert("请先选择一个正在运行的训练任务");
      return;
    }

    const job = jobs.find((item) => item.job_id === selectedJobId);

    if (!job) {
      alert("未找到当前训练任务");
      return;
    }

    if (!["pending", "running"].includes(job.status)) {
      alert("该任务当前状态不可终止");
      return;
    }

    const ok = window.confirm(`确认终止训练任务：${selectedJobId}？`);

    if (!ok) return;

    const result = await terminateTrainJob(selectedJobId);

    if (result.success) {
      alert("终止任务请求已发送");
      await loadJobs();
      await loadTrainLog(selectedJobId);
    } else {
      alert(result.message);
    }
  };

  const handleTerminateJob = async (jobId) => {
    const ok = window.confirm(`确认终止训练任务：${jobId}？`);

    if (!ok) return;

    const result = await terminateTrainJob(jobId);

    if (result.success) {
      alert("终止任务请求已发送");
      await loadJobs();

      if (selectedJobId === jobId) {
        await loadTrainLog(jobId);
      }
    } else {
      alert(result.message);
    }
  };

  const handleDeleteJob = async (jobId) => {
    const ok = window.confirm(`确认删除训练任务记录：${jobId}？`);

    if (!ok) return;

    const result = await deleteTrainJob(jobId, false);

    if (result.success) {
      alert("训练任务记录已删除");

      if (selectedJobId === jobId) {
        setSelectedJobId("");
        setTrainLog("");
      }

      await loadJobs();
    } else {
      alert(result.message);
    }
  };

  const handleToggleModel = async (modelKey, enabled) => {
    const result = await updateModelEnabled(modelKey, enabled);

    if (result.success) {
      await loadModels();
    } else {
      alert(result.message);
    }
  };

  const handleRenameModel = async (model) => {
    setOpenModelMenuKey("");

    const newName = window.prompt("请输入新的模型显示名称", model.label);

    if (!newName || !newName.trim()) return;

    const result = await renameModel(model.key, newName.trim());

    if (result.success) {
      alert("模型重命名成功");
      await loadModels();
    } else {
      alert(result.message);
    }
  };

  const handleDeleteModel = async (model) => {
    setOpenModelMenuKey("");

    const message = model.is_custom
      ? `确认删除自定义模型：${model.label}？\n\n将同时删除模型文件和归一化器文件。`
      : `确认删除内置模型：${model.label}？\n\n内置模型不会删除物理文件，只会从模型管理和用户可调用列表中隐藏。`;

    const ok = window.confirm(message);

    if (!ok) return;

    const result = await deleteModel(model.key, true);

    if (result.success) {
      alert(result.message || "模型删除成功");
      await loadModels();
    } else {
      alert(result.message);
    }
  };

  const handleViewJobLog = async (jobId) => {
    setSelectedJobId(jobId);
    await loadTrainLog(jobId);
  };

  const logout = () => {
    localStorage.removeItem("isLogin");
    localStorage.removeItem("username");
    localStorage.removeItem("role");
    navigate("/admin-login");
  };

  return (
    <div className="home-container">
      <Navbar />

      <main style={styles.page} onClick={() => setOpenModelMenuKey("")}>
        <section style={styles.header}>
          <div>
            <h1 style={styles.title}>管理员后台</h1>
            <p style={styles.subtitle}>
              历史数据集管理、模型训练管理与模型调用权限配置。
            </p>
          </div>

          <button style={styles.logoutButton} onClick={logout}>
            退出登录
          </button>
        </section>

        <section style={styles.tabs}>
          <button
            style={activeTab === "datasets" ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab("datasets")}
          >
            历史数据集CSV管理
          </button>

          <button
            style={activeTab === "training" ? styles.activeTab : styles.tab}
            onClick={() => {
              setActiveTab("training");
              loadJobs();
            }}
          >
            模型训练管理
          </button>

          <button
            style={activeTab === "models" ? styles.activeTab : styles.tab}
            onClick={() => {
              setActiveTab("models");
              loadModels();
            }}
          >
            模型管理
          </button>
        </section>

        {activeTab === "datasets" && (
          <section style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>历史数据集 CSV 管理</h2>

              <div style={styles.actionRow}>
                <label style={styles.uploadButton}>
                  {uploading ? "导入中..." : "导入CSV"}
                  <input
                    type="file"
                    accept=".csv"
                    style={{ display: "none" }}
                    onChange={handleUpload}
                    disabled={uploading}
                  />
                </label>

                <input
                  style={styles.input}
                  value={newDatasetName}
                  placeholder="新建CSV文件名，如 changchun_new.csv"
                  onChange={(e) => setNewDatasetName(e.target.value)}
                />

                <button style={styles.primaryButton} onClick={handleCreateDataset}>
                  新建
                </button>

                <button style={styles.secondaryButton} onClick={loadDatasets}>
                  刷新
                </button>
              </div>
            </div>

            <div style={styles.datasetLayout}>
              <div style={styles.datasetList}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>文件名</th>
                      <th style={styles.th}>行数</th>
                      <th style={styles.th}>最后日期</th>
                      <th style={styles.th}>版本</th>
                      <th style={styles.th}>操作</th>
                    </tr>
                  </thead>

                  <tbody>
                    {datasets.length === 0 ? (
                      <tr>
                        <td style={styles.td} colSpan="5">
                          暂无CSV数据集
                        </td>
                      </tr>
                    ) : (
                      datasets.map((item) => (
                        <tr
                          key={item.filename}
                          style={
                            selectedFile === item.filename
                              ? styles.selectedRow
                              : undefined
                          }
                        >
                          <td style={styles.td}>{item.filename}</td>
                          <td style={styles.td}>{item.rows}</td>
                          <td style={styles.td}>{item.last_date || "--"}</td>
                          <td style={styles.td}>{item.version}</td>
                          <td style={styles.td}>
                            <button
                              style={styles.smallButton}
                              onClick={() => handlePreviewDataset(item.filename)}
                            >
                              预览
                            </button>

                            <button
                              style={styles.smallButton}
                              onClick={() => handleRenameDataset(item.filename)}
                            >
                              重命名
                            </button>

                            <button
                              style={styles.dangerButton}
                              onClick={() => handleDeleteDataset(item.filename)}
                            >
                              删除
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div style={styles.previewPanel}>
                <div style={styles.previewTopBar}>
                  <div style={styles.previewTitleBlock}>
                    <h3 style={styles.previewTitle}>
                      数据预览 {selectedFile ? `：${selectedFile}` : ""}
                    </h3>

                    {selectedDatasetInfo && (
                      <div style={styles.datasetMeta}>
                        <span>
                          字段：{selectedDatasetInfo.columns?.join("，") || "--"}
                        </span>
                        <span>大小：{selectedDatasetInfo.size} bytes</span>

                        {csvEditMode && (
                          <span style={styles.editingTip}>
                            当前为编辑模式，可直接修改单元格、列名、行列结构。
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div style={styles.previewActions}>
                    {!csvEditMode ? (
                      <button style={styles.secondaryButton} onClick={enterCsvEditMode}>
                        进入编辑模式
                      </button>
                    ) : (
                      <>
                        <button style={styles.smallButton} onClick={addCsvRow}>
                          新增行
                        </button>

                        <button style={styles.smallButton} onClick={addCsvColumn}>
                          新增列
                        </button>

                        <button
                          style={styles.primaryButton}
                          onClick={saveCsvEditContent}
                          disabled={savingCsv}
                        >
                          {savingCsv ? "保存中..." : "保存CSV"}
                        </button>

                        <button style={styles.secondaryButton} onClick={cancelCsvEditMode}>
                          取消编辑
                        </button>
                      </>
                    )}
                  </div>
                </div>

                <div style={styles.previewTableOuter}>
                  <div style={styles.previewTableWrap}>
                    {!csvEditMode ? (
                      <table style={styles.previewTable}>
                        <thead>
                          <tr>
                            {preview.columns.map((col) => (
                              <th key={col} style={styles.th}>
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>

                        <tbody>
                          {preview.rows.length === 0 ? (
                            <tr>
                              <td style={styles.td} colSpan={preview.columns.length || 1}>
                                请选择CSV文件进行预览
                              </td>
                            </tr>
                          ) : (
                            preview.rows.map((row, index) => (
                              <tr key={index}>
                                {preview.columns.map((col) => (
                                  <td key={col} style={styles.td}>
                                    {String(row[col] ?? "")}
                                  </td>
                                ))}
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    ) : (
                      <table style={styles.previewTable}>
                        <thead>
                          <tr>
                            {editColumns.map((col, colIndex) => (
                              <th key={colIndex} style={styles.editTh}>
                                <div style={styles.columnEditBox}>
                                  <input
                                    style={styles.columnInput}
                                    value={col}
                                    onChange={(e) =>
                                      changeEditColumnName(colIndex, e.target.value)
                                    }
                                  />

                                  <button
                                    style={styles.deleteColumnButton}
                                    onClick={() => deleteCsvColumn(colIndex)}
                                    type="button"
                                  >
                                    删列
                                  </button>
                                </div>
                              </th>
                            ))}

                            <th style={styles.actionTh}>操作</th>
                          </tr>
                        </thead>

                        <tbody>
                          {editRows.length === 0 ? (
                            <tr>
                              <td style={styles.td} colSpan={editColumns.length + 1}>
                                暂无数据。点击“新增行”开始填写。
                              </td>
                            </tr>
                          ) : (
                            editRows.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                {editColumns.map((col, colIndex) => (
                                  <td key={`${rowIndex}-${colIndex}`} style={styles.editTd}>
                                    <input
                                      style={styles.cellInput}
                                      value={row[colIndex] ?? ""}
                                      onChange={(e) =>
                                        changeEditCell(rowIndex, colIndex, e.target.value)
                                      }
                                    />
                                  </td>
                                ))}

                                <td style={styles.editTd}>
                                  <button
                                    style={styles.dangerButton}
                                    onClick={() => deleteCsvRow(rowIndex)}
                                  >
                                    删除行
                                  </button>
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === "training" && (
          <section style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>模型训练管理</h2>

              <button style={styles.secondaryButton} onClick={loadJobs}>
                刷新训练任务
              </button>
            </div>

            <div style={styles.trainLayout}>
              <div style={styles.trainForm}>
                <h3 style={styles.previewTitle}>训练参数设置</h3>

                <div style={styles.formGrid}>
                  <label style={styles.formItem}>
                    模型名称
                    <input
                      style={styles.inputFull}
                      value={trainForm.model_name}
                      onChange={(e) =>
                        handleTrainFormChange("model_name", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    模型类型
                    <select
                      style={styles.inputFull}
                      value={trainForm.model_key}
                      onChange={(e) =>
                        handleTrainFormChange("model_key", e.target.value)
                      }
                    >
                      {MODEL_OPTIONS.map((item) => (
                        <option key={item.key} value={item.key}>
                          {item.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label style={styles.formItem}>
                    训练轮数 epochs
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.epochs}
                      onChange={(e) =>
                        handleTrainFormChange("epochs", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    batch_size
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.batch_size}
                      onChange={(e) =>
                        handleTrainFormChange("batch_size", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    learning_rate
                    <input
                      type="number"
                      step="0.0001"
                      style={styles.inputFull}
                      value={trainForm.learning_rate}
                      onChange={(e) =>
                        handleTrainFormChange("learning_rate", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    hidden_size
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.hidden_size}
                      onChange={(e) =>
                        handleTrainFormChange("hidden_size", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    num_layers
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.num_layers}
                      onChange={(e) =>
                        handleTrainFormChange("num_layers", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    dropout
                    <input
                      type="number"
                      step="0.1"
                      style={styles.inputFull}
                      value={trainForm.dropout}
                      onChange={(e) =>
                        handleTrainFormChange("dropout", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    seq_length
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.seq_length}
                      onChange={(e) =>
                        handleTrainFormChange("seq_length", e.target.value)
                      }
                    />
                  </label>

                  <label style={styles.formItem}>
                    pred_days
                    <input
                      type="number"
                      style={styles.inputFull}
                      value={trainForm.pred_days}
                      onChange={(e) =>
                        handleTrainFormChange("pred_days", e.target.value)
                      }
                    />
                  </label>
                </div>

                <div style={styles.multiDatasetBox}>
                  <div style={styles.multiDatasetHeader}>
                    <strong>训练数据集</strong>

                    <div>
                      <button style={styles.miniButton} onClick={selectAllDatasets}>
                        全选
                      </button>

                      <button style={styles.miniButton} onClick={clearSelectedDatasets}>
                        清空
                      </button>
                    </div>
                  </div>

                  <div style={styles.multiDatasetList}>
                    {datasets.length === 0 ? (
                      <div style={styles.emptyText}>暂无数据集，请先导入CSV</div>
                    ) : (
                      datasets.map((item) => (
                        <label key={item.filename} style={styles.checkboxRow}>
                          <input
                            type="checkbox"
                            checked={trainForm.dataset_filenames.includes(
                              item.filename
                            )}
                            onChange={() => toggleTrainingDataset(item.filename)}
                          />

                          <span>{item.filename}</span>

                          <small style={styles.datasetSmallText}>
                            {item.last_date
                              ? `最后日期：${item.last_date}`
                              : "日期未知"}
                          </small>
                        </label>
                      ))
                    )}
                  </div>

                  <div style={styles.selectedCountText}>
                    已选择 {trainForm.dataset_filenames.length} 个站点数据集
                  </div>
                </div>

                <label style={styles.formItemWide}>
                  备注
                  <textarea
                    style={styles.textarea}
                    value={trainForm.remark}
                    onChange={(e) => handleTrainFormChange("remark", e.target.value)}
                  />
                </label>

                <div style={styles.trainButtonRow}>
                  <button style={styles.primaryButtonLarge} onClick={handleStartTraining}>
                    提交训练任务
                  </button>

                  <button
                    style={
                      selectedJob && ["pending", "running"].includes(selectedJob.status)
                        ? styles.terminateButtonLarge
                        : styles.disabledButtonLarge
                    }
                    onClick={handleTerminateSelectedJob}
                    disabled={
                      !selectedJob ||
                      !["pending", "running"].includes(selectedJob.status)
                    }
                  >
                    终止任务
                  </button>
                </div>
              </div>

              <div style={styles.jobPanel}>
                <h3 style={styles.previewTitle}>训练任务记录</h3>

                <div style={styles.previewTableWrap}>
                  <table style={styles.table}>
                    <thead>
                      <tr>
                        <th style={styles.th}>任务ID</th>
                        <th style={styles.th}>模型名称</th>
                        <th style={styles.th}>训练集</th>
                        <th style={styles.th}>状态</th>
                        <th style={styles.th}>模型评分</th>
                        <th style={styles.th}>创建时间</th>
                        <th style={styles.th}>操作</th>
                      </tr>
                    </thead>

                    <tbody>
                      {jobs.length === 0 ? (
                        <tr>
                          <td style={styles.td} colSpan="7">
                            暂无训练任务
                          </td>
                        </tr>
                      ) : (
                        jobs.map((job) => (
                          <tr
                            key={job.job_id}
                            style={
                              selectedJobId === job.job_id
                                ? styles.selectedRow
                                : undefined
                            }
                          >
                            <td style={styles.td}>{job.job_id}</td>

                            <td style={styles.td}>{job.model_name}</td>

                            <td style={styles.td}>
                              {Array.isArray(job.dataset_filenames)
                                ? `${job.dataset_filenames.length} 个站点`
                                : job.dataset_filename || "--"}
                            </td>

                            <td style={styles.td}>
                              <span style={getJobStatusStyle(job.status)}>
                                {job.status}
                              </span>

                              {job.error ? (
                                <div style={styles.errorText}>{job.error}</div>
                              ) : null}

                              {job.message ? (
                                <div style={styles.messageText}>{job.message}</div>
                              ) : null}
                            </td>

                            <td style={styles.td}>{job.score_summary || "--"}</td>

                            <td style={styles.td}>{job.created_at || "--"}</td>

                            <td style={styles.td}>
                              <button
                                style={styles.smallButton}
                                onClick={() => handleViewJobLog(job.job_id)}
                              >
                                查看日志
                              </button>

                              {["pending", "running"].includes(job.status) && (
                                <button
                                  style={styles.warningButton}
                                  onClick={() => handleTerminateJob(job.job_id)}
                                >
                                  终止
                                </button>
                              )}

                              <button
                                style={styles.dangerButton}
                                onClick={() => handleDeleteJob(job.job_id)}
                              >
                                删除
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <div style={styles.logPanel}>
                  <div style={styles.logHeader}>
                    <h3 style={styles.logTitle}>
                      训练过程日志 {selectedJob ? `：${selectedJob.job_id}` : ""}
                    </h3>

                    <div style={styles.logTools}>
                      <label style={styles.autoScrollLabel}>
                        <input
                          type="checkbox"
                          checked={autoScrollLog}
                          onChange={(e) => setAutoScrollLog(e.target.checked)}
                        />
                        自动滚动到底部
                      </label>

                      <button
                        style={styles.secondaryButton}
                        onClick={() => selectedJobId && loadTrainLog(selectedJobId)}
                      >
                        {trainLogLoading ? "加载中..." : "刷新日志"}
                      </button>
                    </div>
                  </div>

                  {selectedJob && (
                    <div style={styles.metricBar}>
                      <span>状态：{selectedJob.status}</span>
                      <span>评分：{selectedJob.score_summary || "--"}</span>

                      {selectedJob.metrics?.epoch ? (
                        <span>
                          迭代：{selectedJob.metrics.epoch}
                          {selectedJob.metrics.total_epoch
                            ? ` / ${selectedJob.metrics.total_epoch}`
                            : ""}
                        </span>
                      ) : null}
                    </div>
                  )}

                  <pre ref={logBoxRef} style={styles.logBox}>
                    {trainLog ||
                      "请选择训练任务查看日志。训练脚本需要在每个 epoch 输出 loss / val_loss / rmse / mae / r2 等指标，前端即可实时显示。"}
                  </pre>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === "models" && (
          <section style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>模型管理</h2>

              <button style={styles.secondaryButton} onClick={loadModels}>
                刷新模型状态
              </button>
            </div>

            <div style={styles.modelGrid}>
              {models.length === 0 ? (
                <div style={styles.emptyText}>暂无可管理模型</div>
              ) : (
                models.map((model) => (
                  <div key={model.key} style={styles.modelCard}>
                    <div style={styles.modelMenuWrap}>
                      <button
                        style={styles.modelMenuButton}
                        onClick={(e) => {
                          e.stopPropagation();
                          setOpenModelMenuKey(
                            openModelMenuKey === model.key ? "" : model.key
                          );
                        }}
                      >
                        ⋯
                      </button>

                      {openModelMenuKey === model.key && (
                        <div
                          style={styles.modelMenu}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            style={styles.modelMenuItem}
                            onClick={() => handleRenameModel(model)}
                          >
                            重命名
                          </button>

                          <button
                            style={styles.modelMenuDangerItem}
                            onClick={() => handleDeleteModel(model)}
                          >
                            删除
                          </button>
                        </div>
                      )}
                    </div>

                    <div style={styles.modelCardHeader}>
                      <h3 style={styles.modelTitle}>{model.label}</h3>

                      <label style={styles.switchLabel}>
                        <input
                          type="checkbox"
                          checked={model.enabled}
                          onChange={(e) =>
                            handleToggleModel(model.key, e.target.checked)
                          }
                        />
                        允许用户调用
                      </label>
                    </div>

                    <p style={styles.modelDesc}>{model.description}</p>

                    <div style={styles.modelMeta}>
                      <span>
                        模型类型：
                        {model.is_custom
                          ? `自定义模型（基于 ${model.base_model_label || "--"}）`
                          : "内置模型"}
                      </span>

                      {model.created_at ? (
                        <span>创建时间：{model.created_at}</span>
                      ) : null}

                      <span>模型文件：{model.model_file}</span>

                      <span>
                        模型文件状态：
                        <strong style={model.model_exists ? styles.okText : styles.badText}>
                          {model.model_exists ? "存在" : "缺失"}
                        </strong>
                      </span>

                      <span>归一化器：{model.scaler_file}</span>

                      <span>
                        归一化器状态：
                        <strong style={model.scaler_exists ? styles.okText : styles.badText}>
                          {model.scaler_exists ? "存在" : "缺失"}
                        </strong>
                      </span>

                      <span>
                        用户可调用状态：
                        <strong style={model.available ? styles.okText : styles.badText}>
                          {model.available ? "可调用" : "不可调用"}
                        </strong>
                      </span>

                      {model.hidden_size ? (
                        <span>hidden_size：{model.hidden_size}</span>
                      ) : null}

                      {model.seq_length ? (
                        <span>seq_length：{model.seq_length}</span>
                      ) : null}

                      {model.pred_days ? (
                        <span>pred_days：{model.pred_days}</span>
                      ) : null}

                      {model.metrics?.test_rmse ? (
                        <span>
                          评分：Test RMSE={model.metrics.test_rmse}
                          {model.metrics.test_mae
                            ? `，Test MAE=${model.metrics.test_mae}`
                            : ""}
                          {model.metrics.test_r2
                            ? `，Test R²=${model.metrics.test_r2}`
                            : ""}
                        </span>
                      ) : null}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
}

function getJobStatusStyle(status) {
  if (status === "finished") {
    return {
      color: "#4ade80",
      fontWeight: "bold",
    };
  }

  if (status === "running") {
    return {
      color: "#00ffff",
      fontWeight: "bold",
    };
  }

  if (status === "failed") {
    return {
      color: "#ef4444",
      fontWeight: "bold",
    };
  }

  if (status === "terminated") {
    return {
      color: "#facc15",
      fontWeight: "bold",
    };
  }

  return {
    color: "#facc15",
    fontWeight: "bold",
  };
}

const styles = {
  page: {
    flex: 1,
    width: "100%",
    padding: "110px 40px 50px",
    color: "#ffffff",
    overflowX: "hidden",
  },

  header: {
    maxWidth: "1280px",
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

  tabs: {
    maxWidth: "1280px",
    margin: "0 auto 24px",
    padding: "12px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.55)",
    display: "flex",
    gap: "12px",
    flexWrap: "wrap",
  },

  tab: {
    padding: "12px 20px",
    border: "1px solid rgba(0,255,255,0.25)",
    borderRadius: "8px",
    background: "rgba(0,0,0,0.35)",
    color: "#00ffff",
    cursor: "pointer",
    fontWeight: "bold",
  },

  activeTab: {
    padding: "12px 20px",
    border: "1px solid rgba(0,255,255,0.8)",
    borderRadius: "8px",
    background: "linear-gradient(90deg, #00bfff, #007fff)",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: "bold",
  },

  panel: {
    maxWidth: "1280px",
    width: "100%",
    margin: "0 auto",
    padding: "24px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.6)",
    overflow: "hidden",
  },

  panelHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "16px",
    flexWrap: "wrap",
    marginBottom: "20px",
  },

  panelTitle: {
    margin: 0,
    color: "#ffffff",
    fontSize: "24px",
  },

  actionRow: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    flexWrap: "wrap",
    maxWidth: "100%",
  },

  uploadButton: {
    padding: "10px 18px",
    borderRadius: "6px",
    background: "linear-gradient(90deg, #00bfff, #007fff)",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: "bold",
    whiteSpace: "nowrap",
  },

  input: {
    height: "40px",
    minWidth: "260px",
    padding: "0 12px",
    borderRadius: "6px",
    border: "none",
    outline: "none",
  },

  primaryButton: {
    height: "40px",
    padding: "0 18px",
    border: "none",
    borderRadius: "6px",
    background: "linear-gradient(90deg, #00bfff, #007fff)",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: "bold",
    whiteSpace: "nowrap",
  },

  primaryButtonLarge: {
    width: "100%",
    height: "44px",
    border: "none",
    borderRadius: "6px",
    background: "linear-gradient(90deg, #00bfff, #007fff)",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: "bold",
    fontSize: "16px",
  },

  terminateButtonLarge: {
    width: "100%",
    height: "44px",
    border: "1px solid rgba(250,204,21,0.75)",
    borderRadius: "6px",
    background: "rgba(120,53,15,0.75)",
    color: "#fde68a",
    cursor: "pointer",
    fontWeight: "bold",
    fontSize: "16px",
  },

  disabledButtonLarge: {
    width: "100%",
    height: "44px",
    border: "1px solid rgba(255,255,255,0.18)",
    borderRadius: "6px",
    background: "rgba(255,255,255,0.08)",
    color: "rgba(255,255,255,0.35)",
    cursor: "not-allowed",
    fontWeight: "bold",
    fontSize: "16px",
  },

  trainButtonRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "12px",
    marginTop: "18px",
  },

  secondaryButton: {
    height: "40px",
    padding: "0 18px",
    border: "1px solid rgba(0,255,255,0.45)",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.35)",
    color: "#00ffff",
    cursor: "pointer",
    fontWeight: "bold",
    whiteSpace: "nowrap",
  },

  miniButton: {
    marginLeft: "8px",
    padding: "4px 10px",
    border: "1px solid rgba(0,255,255,0.45)",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.35)",
    color: "#00ffff",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  datasetLayout: {
    display: "grid",
    gridTemplateColumns: "1fr",
    gap: "20px",
    maxWidth: "100%",
    overflow: "hidden",
  },

  datasetList: {
    width: "100%",
    maxWidth: "100%",
    overflowX: "auto",
  },

  previewPanel: {
    width: "100%",
    maxWidth: "100%",
    padding: "20px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.38)",
    overflow: "hidden",
  },

  previewTopBar: {
    width: "100%",
    maxWidth: "100%",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: "16px",
    flexWrap: "wrap",
    marginBottom: "16px",
  },

  previewTitleBlock: {
    flex: "1 1 420px",
    minWidth: 0,
  },

  previewActions: {
    flex: "1 1 360px",
    maxWidth: "100%",
    display: "flex",
    justifyContent: "flex-end",
    alignItems: "center",
    gap: "10px",
    flexWrap: "wrap",
  },

  previewTitle: {
    margin: "0 0 14px",
    color: "#00ffff",
    fontSize: "20px",
    wordBreak: "break-word",
  },

  datasetMeta: {
    display: "flex",
    gap: "18px",
    flexWrap: "wrap",
    color: "rgba(255,255,255,0.75)",
    marginBottom: "4px",
    lineHeight: 1.6,
  },

  editingTip: {
    color: "#facc15",
    fontWeight: "bold",
  },

  previewTableOuter: {
    width: "100%",
    maxWidth: "100%",
    overflow: "hidden",
    borderRadius: "8px",
    border: "1px solid rgba(255,255,255,0.12)",
  },

  previewTableWrap: {
    width: "100%",
    maxWidth: "100%",
    maxHeight: "430px",
    overflowX: "auto",
    overflowY: "auto",
  },

  table: {
    width: "100%",
    minWidth: "1020px",
    borderCollapse: "collapse",
  },

  previewTable: {
    width: "max-content",
    minWidth: "100%",
    borderCollapse: "collapse",
    tableLayout: "auto",
  },

  th: {
    padding: "12px",
    textAlign: "left",
    color: "#00ffff",
    background: "rgba(0,0,0,0.52)",
    borderBottom: "1px solid rgba(255,255,255,0.25)",
    whiteSpace: "nowrap",
  },

  td: {
    padding: "12px",
    borderBottom: "1px solid rgba(255,255,255,0.12)",
    color: "rgba(255,255,255,0.86)",
    whiteSpace: "nowrap",
  },

  editTh: {
    padding: "10px",
    textAlign: "left",
    color: "#00ffff",
    background: "rgba(0,0,0,0.52)",
    borderBottom: "1px solid rgba(255,255,255,0.25)",
    whiteSpace: "nowrap",
    minWidth: "190px",
  },

  actionTh: {
    padding: "10px",
    textAlign: "center",
    color: "#00ffff",
    background: "rgba(0,0,0,0.52)",
    borderBottom: "1px solid rgba(255,255,255,0.25)",
    whiteSpace: "nowrap",
    minWidth: "120px",
  },

  editTd: {
    padding: "8px 10px",
    borderBottom: "1px solid rgba(255,255,255,0.12)",
    color: "rgba(255,255,255,0.86)",
    whiteSpace: "nowrap",
  },

  columnEditBox: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },

  columnInput: {
    width: "130px",
    height: "34px",
    padding: "0 8px",
    border: "1px solid rgba(0,255,255,0.35)",
    borderRadius: "6px",
    background: "rgba(255,255,255,0.95)",
    color: "#111827",
    outline: "none",
  },

  cellInput: {
    width: "160px",
    height: "36px",
    padding: "0 10px",
    border: "1px solid rgba(255,255,255,0.22)",
    borderRadius: "6px",
    background: "rgba(255,255,255,0.95)",
    color: "#111827",
    outline: "none",
  },

  deleteColumnButton: {
    height: "30px",
    padding: "0 10px",
    border: "1px solid rgba(239,68,68,0.75)",
    borderRadius: "6px",
    background: "rgba(127,29,29,0.55)",
    color: "#fecaca",
    cursor: "pointer",
    fontWeight: "bold",
    whiteSpace: "nowrap",
  },

  selectedRow: {
    background: "rgba(0,255,255,0.12)",
  },

  smallButton: {
    padding: "8px 12px",
    border: "1px solid rgba(0,255,255,0.45)",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.35)",
    color: "#00ffff",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  warningButton: {
    marginRight: "8px",
    padding: "6px 10px",
    border: "1px solid rgba(250,204,21,0.75)",
    borderRadius: "6px",
    background: "rgba(120,53,15,0.55)",
    color: "#fde68a",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  dangerButton: {
    padding: "6px 10px",
    border: "1px solid rgba(239,68,68,0.75)",
    borderRadius: "6px",
    background: "rgba(127,29,29,0.45)",
    color: "#fecaca",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  trainLayout: {
    display: "grid",
    gridTemplateColumns: "minmax(360px, 460px) 1fr",
    gap: "22px",
  },

  trainForm: {
    padding: "20px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.38)",
  },

  formGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "14px",
  },

  formItem: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    color: "#00ffff",
    fontWeight: "bold",
  },

  formItemWide: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    marginTop: "14px",
    color: "#00ffff",
    fontWeight: "bold",
  },

  inputFull: {
    width: "100%",
    height: "38px",
    padding: "0 10px",
    border: "none",
    borderRadius: "6px",
    outline: "none",
  },

  textarea: {
    width: "100%",
    minHeight: "80px",
    padding: "10px",
    border: "none",
    borderRadius: "6px",
    outline: "none",
    resize: "vertical",
  },

  multiDatasetBox: {
    marginTop: "16px",
    padding: "14px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.35)",
    border: "1px solid rgba(255,255,255,0.12)",
  },

  multiDatasetHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    color: "#00ffff",
    marginBottom: "10px",
  },

  multiDatasetList: {
    maxHeight: "220px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },

  checkboxRow: {
    display: "grid",
    gridTemplateColumns: "22px 1fr auto",
    alignItems: "center",
    gap: "8px",
    color: "#ffffff",
    padding: "8px",
    borderRadius: "6px",
    background: "rgba(255,255,255,0.05)",
  },

  datasetSmallText: {
    color: "rgba(255,255,255,0.65)",
    fontSize: "12px",
  },

  selectedCountText: {
    marginTop: "10px",
    color: "#facc15",
    fontSize: "13px",
  },

  emptyText: {
    color: "rgba(255,255,255,0.65)",
    padding: "10px",
  },

  jobPanel: {
    padding: "20px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.38)",
    overflowX: "auto",
  },

  errorText: {
    marginTop: "4px",
    color: "#fecaca",
    fontSize: "12px",
    whiteSpace: "normal",
  },

  messageText: {
    marginTop: "4px",
    color: "rgba(255,255,255,0.72)",
    fontSize: "12px",
    whiteSpace: "normal",
  },

  logPanel: {
    marginTop: "22px",
    padding: "16px",
    borderRadius: "10px",
    background: "rgba(0,0,0,0.42)",
    border: "1px solid rgba(255,255,255,0.12)",
  },

  logHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "12px",
    marginBottom: "12px",
    flexWrap: "wrap",
  },

  logTools: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    flexWrap: "wrap",
  },

  autoScrollLabel: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    color: "rgba(255,255,255,0.82)",
    fontSize: "14px",
    cursor: "pointer",
    userSelect: "none",
  },

  logTitle: {
    margin: 0,
    color: "#00ffff",
    fontSize: "18px",
  },

  metricBar: {
    display: "flex",
    flexWrap: "wrap",
    gap: "14px",
    marginBottom: "12px",
    color: "rgba(255,255,255,0.82)",
  },

  logBox: {
    margin: 0,
    padding: "14px",
    height: "320px",
    overflow: "auto",
    borderRadius: "8px",
    background: "#071018",
    color: "#d1fae5",
    fontSize: "13px",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
  },

  modelGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
    gap: "20px",
  },

  modelCard: {
    position: "relative",
    padding: "20px",
    borderRadius: "12px",
    background: "rgba(0,0,0,0.42)",
    border: "1px solid rgba(255,255,255,0.12)",
  },

  modelMenuWrap: {
    position: "absolute",
    top: "12px",
    right: "12px",
    zIndex: 5,
  },

  modelMenuButton: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    border: "1px solid rgba(0,255,255,0.35)",
    background: "rgba(0,0,0,0.45)",
    color: "#00ffff",
    cursor: "pointer",
    fontSize: "20px",
    lineHeight: "24px",
  },

  modelMenu: {
    position: "absolute",
    top: "38px",
    right: 0,
    minWidth: "120px",
    padding: "6px",
    borderRadius: "8px",
    background: "rgba(5,15,20,0.98)",
    border: "1px solid rgba(0,255,255,0.25)",
    boxShadow: "0 8px 22px rgba(0,0,0,0.35)",
  },

  modelMenuItem: {
    width: "100%",
    padding: "9px 12px",
    border: "none",
    borderRadius: "6px",
    background: "transparent",
    color: "#00ffff",
    cursor: "pointer",
    textAlign: "left",
  },

  modelMenuDangerItem: {
    width: "100%",
    padding: "9px 12px",
    border: "none",
    borderRadius: "6px",
    background: "transparent",
    color: "#fecaca",
    cursor: "pointer",
    textAlign: "left",
  },

  modelCardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "12px",
    paddingRight: "34px",
  },

  modelTitle: {
    margin: 0,
    color: "#00ffff",
    fontSize: "20px",
  },

  switchLabel: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "#ffffff",
    whiteSpace: "nowrap",
  },

  modelDesc: {
    color: "rgba(255,255,255,0.75)",
    lineHeight: 1.6,
  },

  modelMeta: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    color: "rgba(255,255,255,0.82)",
  },

  okText: {
    color: "#4ade80",
    marginLeft: "6px",
  },

  badText: {
    color: "#ef4444",
    marginLeft: "6px",
  },
};