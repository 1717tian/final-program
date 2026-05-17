import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 30000,
});

export const login = async (data) => {
  try {
    const res = await API.post("/api/login", data);
    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "登录失败",
    };
  }
};

export const register = async (data) => {
  try {
    const res = await API.post("/api/register", data);
    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "注册失败",
    };
  }
};

export const adminLogin = async (data) => {
  try {
    const res = await API.post("/api/login", data);

    if (res.data.role !== "admin") {
      return {
        success: false,
        message: "当前账号不是管理员账号",
      };
    }

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "管理员登录失败",
    };
  }
};

export const fetchModels = async () => {
  try {
    const res = await API.get("/models");
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "模型列表加载失败",
      models: [],
    };
  }
};

export const fetchStations = async () => {
  try {
    const res = await API.get("/stations");
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "站点加载失败",
      stations: [],
    };
  }
};

export const fetchPrediction = async (data) => {
  try {
    const res = await API.post("/predict", data);
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "预测失败",
    };
  }
};

export const fetchDatasets = async () => {
  try {
    const res = await API.get("/admin/datasets");
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "数据集加载失败",
      datasets: [],
    };
  }
};

export const uploadDataset = async (formData) => {
  try {
    const res = await API.post("/admin/datasets/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV导入失败",
    };
  }
};

export const createDataset = async (data) => {
  try {
    const res = await API.post("/admin/datasets/create", data);
    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV新建失败",
    };
  }
};

export const renameDataset = async (filename, newFilename) => {
  try {
    const res = await API.put(
      `/admin/datasets/${encodeURIComponent(filename)}/rename`,
      {
        new_filename: newFilename,
      }
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV重命名失败",
    };
  }
};

export const deleteDataset = async (filename) => {
  try {
    const res = await API.delete(
      `/admin/datasets/${encodeURIComponent(filename)}`
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV删除失败",
    };
  }
};

export const previewDataset = async (filename, limit = 100) => {
  try {
    const res = await API.get(
      `/admin/datasets/${encodeURIComponent(filename)}/preview`,
      {
        params: {
          limit,
        },
      }
    );

    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV预览失败",
      columns: [],
      rows: [],
    };
  }
};

export const saveDatasetContent = async (filename, payload) => {
  try {
    const res = await API.put(
      `/admin/datasets/${encodeURIComponent(filename)}/content`,
      payload
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "CSV保存失败",
    };
  }
};

export const uploadCsv = uploadDataset;

export const startTraining = async (data) => {
  try {
    const res = await API.post("/admin/train", data);

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "训练任务提交失败",
    };
  }
};

export const fetchTrainJobs = async () => {
  try {
    const res = await API.get("/admin/train/jobs");
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "训练任务加载失败",
      jobs: [],
    };
  }
};

export const fetchTrainJobLog = async (jobId, tailLines = 300) => {
  try {
    const res = await API.get(
      `/admin/train/jobs/${encodeURIComponent(jobId)}/log`,
      {
        params: {
          tail_lines: tailLines,
        },
      }
    );

    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "训练日志加载失败",
      log: "",
      metrics: {},
      score_summary: "--",
    };
  }
};

export const terminateTrainJob = async (jobId) => {
  try {
    const res = await API.post(
      `/admin/train/jobs/${encodeURIComponent(jobId)}/terminate`
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "终止训练任务失败",
    };
  }
};

export const deleteTrainJob = async (jobId, deleteLog = false) => {
  try {
    const res = await API.delete(
      `/admin/train/jobs/${encodeURIComponent(jobId)}`,
      {
        params: {
          delete_log: deleteLog,
        },
      }
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "删除训练任务记录失败",
    };
  }
};

export const fetchAdminModels = async () => {
  try {
    const res = await API.get("/admin/models");
    return res.data;
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "模型管理列表加载失败",
      models: [],
    };
  }
};

export const backfillModelParams = async () => {
  try {
    const res = await API.post("/admin/models/backfill-params");

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "旧模型参数补全失败",
    };
  }
};

export const updateModelEnabled = async (modelKey, enabled) => {
  try {
    const res = await API.put(
      `/admin/models/${encodeURIComponent(modelKey)}/enabled`,
      {
        enabled,
      }
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "模型权限更新失败",
    };
  }
};

export const renameModel = async (modelKey, newLabel) => {
  try {
    const res = await API.put(
      `/admin/models/${encodeURIComponent(modelKey)}/rename`,
      {
        new_label: newLabel,
      }
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "模型重命名失败",
    };
  }
};

export const deleteModel = async (modelKey, deleteFiles = true) => {
  try {
    const res = await API.delete(
      `/admin/models/${encodeURIComponent(modelKey)}`,
      {
        params: {
          delete_files: deleteFiles,
        },
      }
    );

    return {
      success: true,
      ...res.data,
    };
  } catch (err) {
    return {
      success: false,
      message: err.response?.data?.detail || err.message || "模型删除失败",
    };
  }
};

export default API;