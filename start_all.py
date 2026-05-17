import os
import sys
import time
import socket
import shutil
import queue
import threading
import subprocess
import webbrowser
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext


# =========================================================
# PM2.5 监控预警平台 可视化启动器
#
# 新增能力：
# 1. 启动前依赖检查
# 2. 缺少依赖时弹出下载地址和部署提示
# 3. 支持数据库配置保存到 .env
# 4. 支持 PostgreSQL 连接验证
# 5. 支持 PyInstaller 打包为 exe 后作为项目启动器
# =========================================================


BACKEND_PORT = 8000
FRONTEND_PORT = 5173

BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

APP_TITLE = "PM2.5 监控预警平台启动器"

PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/"
NODE_DOWNLOAD_URL = "https://nodejs.org/en/download"
POSTGRESQL_DOWNLOAD_URL = "https://www.postgresql.org/download/windows/"
GIT_DOWNLOAD_URL = "https://git-scm.com/install/windows"

PYTHON_IMPORT_CHECKS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pandas": "pandas",
    "numpy": "numpy",
    "torch": "torch",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "psycopg2": "psycopg2-binary",
    "pydantic": "pydantic",
    "multipart": "python-multipart",
}

PIP_INSTALL_FALLBACK = (
    "python -m pip install fastapi uvicorn pandas numpy scikit-learn "
    "joblib torch psycopg2-binary pydantic python-multipart python-dotenv"
)


def is_windows() -> bool:
    return os.name == "nt"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def get_app_base_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def find_project_root() -> Path:
    start_dir = get_app_base_dir()
    candidates = [start_dir, *start_dir.parents]

    for path in candidates:
        if (path / "backend").exists() and (path / "frontend").exists():
            return path

    raise RuntimeError(
        "未找到项目根目录。\n\n"
        "请确认启动器位于项目目录内，并且项目目录下存在：\n"
        "backend/ 和 frontend/ 两个文件夹。\n\n"
        "如果启动器已经打包为 exe，请把 exe 放在项目根目录。"
    )


PROJECT_ROOT = find_project_root()
BACKEND_DIR = PROJECT_ROOT
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENV_FILE = PROJECT_ROOT / ".env"


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)

        try:
            return sock.connect_ex((host, int(port))) == 0
        except Exception:
            return False


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_port_open(host, port):
            return True

        time.sleep(0.4)

    return False


def run_command_capture(cmd, cwd=None, env=None, timeout=20):
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

        return result.returncode, result.stdout.strip()

    except subprocess.TimeoutExpired:
        return 124, "命令执行超时"
    except Exception as exc:
        return 1, str(exc)


def get_python_command() -> str:
    """
    优先使用项目内 .venv1 的 Python。
    打包成 exe 后，不能直接用 sys.executable 启动后端，因为 sys.executable 会变成启动器自身。
    """

    venv_python = PROJECT_ROOT / ".venv1" / "Scripts" / "python.exe"

    if venv_python.exists():
        return str(venv_python)

    python_cmd = shutil.which("python")

    if python_cmd:
        return python_cmd

    if not is_frozen():
        return sys.executable

    raise RuntimeError(
        "未找到 Python 解释器。\n\n"
        "建议安装 Python 3.10+，并勾选 Add Python to PATH。\n"
        "也可以在项目根目录创建虚拟环境：.venv1"
    )


def get_npm_command() -> str:
    if is_windows():
        npm_cmd = shutil.which("npm.cmd")
    else:
        npm_cmd = shutil.which("npm")

    if not npm_cmd:
        raise RuntimeError(
            "未找到 npm。\n\n"
            "请先安装 Node.js LTS，并确认 npm 已加入系统 PATH。"
        )

    return npm_cmd


def taskkill_pid(pid: int):
    if not is_windows():
        return

    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def read_env_file(path: Path) -> dict:
    data = {}

    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")

    return data


def write_env_file(path: Path, values: dict):
    preserved_lines = []
    existing = read_env_file(path)

    existing.update(values)

    ordered_keys = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    for key in ordered_keys:
        preserved_lines.append(f"{key}={existing.get(key, '')}")

    path.write_text("\n".join(preserved_lines) + "\n", encoding="utf-8")


class DependencyIssue:
    def __init__(
        self,
        name: str,
        message: str,
        fix: str,
        url: str = "",
        severity: str = "ERROR",
    ):
        self.name = name
        self.message = message
        self.fix = fix
        self.url = url
        self.severity = severity

    @property
    def blocking(self) -> bool:
        return self.severity.upper() == "ERROR"


class ServiceProcess:
    def __init__(self, name, port, url):
        self.name = name
        self.port = port
        self.url = url
        self.process = None
        self.external_running = False

    @property
    def pid(self):
        if self.process and self.process.poll() is None:
            return self.process.pid

        return None

    @property
    def running(self):
        if self.process and self.process.poll() is None:
            return True

        if self.external_running and is_port_open("127.0.0.1", self.port):
            return True

        return False


class TechLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x840")
        self.root.minsize(1120, 760)
        self.root.configure(bg="#06111f")

        self.log_queue = queue.Queue()

        self.backend = ServiceProcess("后端 FastAPI", BACKEND_PORT, BACKEND_URL)
        self.frontend = ServiceProcess("前端 Vite", FRONTEND_PORT, FRONTEND_URL)

        self.starting = False
        self.stopping = False
        self.auto_open_browser = tk.BooleanVar(value=True)

        self.backend_status_var = tk.StringVar(value="未启动")
        self.frontend_status_var = tk.StringVar(value="未启动")

        self.backend_pid_var = tk.StringVar(value="--")
        self.frontend_pid_var = tk.StringVar(value="--")

        self.backend_port_var = tk.StringVar(value=str(BACKEND_PORT))
        self.frontend_port_var = tk.StringVar(value=str(FRONTEND_PORT))

        self.system_status_var = tk.StringVar(value="待启动")

        env_data = read_env_file(ENV_FILE)

        self.db_host_var = tk.StringVar(value=env_data.get("DB_HOST", "localhost"))
        self.db_port_var = tk.StringVar(value=env_data.get("DB_PORT", "5432"))
        self.db_name_var = tk.StringVar(value=env_data.get("DB_NAME", "db1"))
        self.db_user_var = tk.StringVar(value=env_data.get("DB_USER", "postgres"))
        self.db_password_var = tk.StringVar(value=env_data.get("DB_PASSWORD", ""))

        self._build_ui()
        self._draw_grid()
        self._poll_log_queue()
        self._refresh_status_loop()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =====================================================
    # UI 构建
    # =====================================================

    def _build_ui(self):
        self.bg_canvas = tk.Canvas(
            self.root,
            bg="#06111f",
            highlightthickness=0,
        )
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        self.main = tk.Frame(self.root, bg="#06111f")
        self.main.place(x=0, y=0, relwidth=1, relheight=1)

        self._build_header()
        self._build_status_area()
        self._build_db_area()
        self._build_control_area()
        self._build_log_area()
        self._build_footer()

    def _build_header(self):
        header = tk.Frame(self.main, bg="#06111f")
        header.pack(fill="x", padx=28, pady=(22, 10))

        left = tk.Frame(header, bg="#06111f")
        left.pack(side="left", fill="x", expand=True)

        title = tk.Label(
            left,
            text="PM2.5 监控预警平台",
            fg="#36f7ff",
            bg="#06111f",
            font=("Microsoft YaHei UI", 26, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            left,
            text="FastAPI 后端 · React/Vite 前端 · PostgreSQL · LSTM PM2.5 预测与预警系统",
            fg="#9eefff",
            bg="#06111f",
            font=("Microsoft YaHei UI", 11),
        )
        subtitle.pack(anchor="w", pady=(6, 0))

        path_label = tk.Label(
            left,
            text=f"项目目录：{PROJECT_ROOT}",
            fg="#6ea8b5",
            bg="#06111f",
            font=("Consolas", 9),
        )
        path_label.pack(anchor="w", pady=(4, 0))

        right = tk.Frame(header, bg="#06111f")
        right.pack(side="right")

        status_box = tk.Frame(
            right,
            bg="#092235",
            highlightbackground="#0ff",
            highlightthickness=1,
        )
        status_box.pack()

        tk.Label(
            status_box,
            text="SYSTEM STATUS",
            fg="#36f7ff",
            bg="#092235",
            font=("Consolas", 10, "bold"),
        ).pack(padx=18, pady=(10, 2))

        tk.Label(
            status_box,
            textvariable=self.system_status_var,
            fg="#ffffff",
            bg="#092235",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(padx=18, pady=(0, 10))

    def _build_status_area(self):
        area = tk.Frame(self.main, bg="#06111f")
        area.pack(fill="x", padx=28, pady=8)

        self.backend_card = self._create_service_card(
            area,
            title="后端服务",
            subtitle="FastAPI · Uvicorn · PostgreSQL · LSTM 模型接口",
            status_var=self.backend_status_var,
            pid_var=self.backend_pid_var,
            port_var=self.backend_port_var,
            url=BACKEND_URL,
            accent="#36f7ff",
        )
        self.backend_card.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.frontend_card = self._create_service_card(
            area,
            title="前端服务",
            subtitle="React · Vite · 可视化预测页面 · 管理员后台",
            status_var=self.frontend_status_var,
            pid_var=self.frontend_pid_var,
            port_var=self.frontend_port_var,
            url=FRONTEND_URL,
            accent="#7cffb2",
        )
        self.frontend_card.pack(side="left", fill="x", expand=True, padx=(10, 0))

    def _create_service_card(
        self,
        parent,
        title,
        subtitle,
        status_var,
        pid_var,
        port_var,
        url,
        accent,
    ):
        card = tk.Frame(
            parent,
            bg="#0a1b2c",
            highlightbackground="#17394f",
            highlightthickness=1,
        )

        top = tk.Frame(card, bg="#0a1b2c")
        top.pack(fill="x", padx=18, pady=(14, 6))

        dot = tk.Canvas(top, width=14, height=14, bg="#0a1b2c", highlightthickness=0)
        dot.pack(side="left", padx=(0, 8))
        dot.create_oval(2, 2, 12, 12, fill=accent, outline=accent)

        tk.Label(
            top,
            text=title,
            fg=accent,
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side="left")

        tk.Label(
            card,
            text=subtitle,
            fg="#b7d6e3",
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", padx=18)

        info = tk.Frame(card, bg="#0a1b2c")
        info.pack(fill="x", padx=18, pady=10)

        self._info_item(info, "状态", status_var, 0, accent)
        self._info_item(info, "PID", pid_var, 1, accent)
        self._info_item(info, "端口", port_var, 2, accent)

        tk.Label(
            card,
            text=url,
            fg="#d9fbff",
            bg="#092235",
            font=("Consolas", 11, "bold"),
            padx=12,
            pady=8,
        ).pack(fill="x", padx=18, pady=(0, 14))

        return card

    def _info_item(self, parent, label, var, col, accent):
        box = tk.Frame(parent, bg="#092235")
        box.grid(row=0, column=col, sticky="ew", padx=5)

        parent.grid_columnconfigure(col, weight=1)

        tk.Label(
            box,
            text=label,
            fg="#82a9b8",
            bg="#092235",
            font=("Microsoft YaHei UI", 9),
        ).pack(pady=(7, 2))

        tk.Label(
            box,
            textvariable=var,
            fg=accent,
            bg="#092235",
            font=("Consolas", 12, "bold"),
        ).pack(pady=(0, 7))

    def _build_db_area(self):
        area = tk.Frame(
            self.main,
            bg="#0a1b2c",
            highlightbackground="#17394f",
            highlightthickness=1,
        )
        area.pack(fill="x", padx=28, pady=(2, 10))

        header = tk.Frame(area, bg="#0a1b2c")
        header.pack(fill="x", padx=16, pady=(10, 6))

        tk.Label(
            header,
            text="数据库配置",
            fg="#36f7ff",
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")

        tk.Label(
            header,
            text=f"配置文件：{ENV_FILE}",
            fg="#7a9cab",
            bg="#0a1b2c",
            font=("Consolas", 9),
        ).pack(side="left", padx=14)

        form = tk.Frame(area, bg="#0a1b2c")
        form.pack(fill="x", padx=14, pady=(0, 12))

        self._db_input(form, "主机", self.db_host_var, 0, width=16)
        self._db_input(form, "端口", self.db_port_var, 1, width=8)
        self._db_input(form, "数据库", self.db_name_var, 2, width=14)
        self._db_input(form, "用户名", self.db_user_var, 3, width=14)
        self._db_input(form, "密码", self.db_password_var, 4, width=18, show="*")

        save_btn = self._button(
            form,
            "保存配置",
            self.save_db_config,
            bg="#123b58",
            fg="#d9fbff",
            width=10,
            height=1,
        )
        save_btn.grid(row=0, column=5, padx=(14, 6), sticky="ew")

        check_btn = self._button(
            form,
            "验证数据库",
            self.validate_database_button,
            bg="#097a47",
            fg="#eafff2",
            width=12,
            height=1,
        )
        check_btn.grid(row=0, column=6, padx=6, sticky="ew")

    def _db_input(self, parent, label, var, col, width=12, show=None):
        box = tk.Frame(parent, bg="#0a1b2c")
        box.grid(row=0, column=col, sticky="ew", padx=4)

        tk.Label(
            box,
            text=label,
            fg="#82a9b8",
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w")

        tk.Entry(
            box,
            textvariable=var,
            width=width,
            show=show,
            bg="#030b12",
            fg="#d9fbff",
            insertbackground="#36f7ff",
            relief="flat",
            font=("Consolas", 10),
        ).pack(fill="x", pady=(4, 0), ipady=4)

    def _build_control_area(self):
        area = tk.Frame(self.main, bg="#06111f")
        area.pack(fill="x", padx=28, pady=(0, 12))

        left = tk.Frame(area, bg="#06111f")
        left.pack(side="left", fill="x", expand=True)

        self.start_btn = self._button(
            left,
            "启动全部",
            self.start_all,
            bg="#0884ff",
            fg="#ffffff",
            width=12,
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = self._button(
            left,
            "停止全部",
            self.stop_all,
            bg="#802020",
            fg="#ffdede",
            width=12,
        )
        self.stop_btn.pack(side="left", padx=8)

        self.restart_btn = self._button(
            left,
            "重启全部",
            self.restart_all,
            bg="#0c4f6f",
            fg="#c8f7ff",
            width=12,
        )
        self.restart_btn.pack(side="left", padx=8)

        self.open_btn = self._button(
            left,
            "打开系统",
            self.open_frontend,
            bg="#097a47",
            fg="#eafff2",
            width=12,
        )
        self.open_btn.pack(side="left", padx=8)

        self.check_btn = self._button(
            left,
            "环境检查",
            self.check_environment,
            bg="#1b3348",
            fg="#36f7ff",
            width=12,
        )
        self.check_btn.pack(side="left", padx=8)

        self.dep_btn = self._button(
            left,
            "依赖检查",
            self.check_dependencies_button,
            bg="#45326b",
            fg="#efe7ff",
            width=12,
        )
        self.dep_btn.pack(side="left", padx=8)

        right = tk.Frame(area, bg="#06111f")
        right.pack(side="right")

        tk.Checkbutton(
            right,
            text="启动后自动打开浏览器",
            variable=self.auto_open_browser,
            fg="#b7d6e3",
            bg="#06111f",
            selectcolor="#0a1b2c",
            activebackground="#06111f",
            activeforeground="#36f7ff",
            font=("Microsoft YaHei UI", 10),
        ).pack(side="right")

    def _build_log_area(self):
        area = tk.Frame(
            self.main,
            bg="#0a1b2c",
            highlightbackground="#17394f",
            highlightthickness=1,
        )
        area.pack(fill="both", expand=True, padx=28, pady=(0, 12))

        header = tk.Frame(area, bg="#0a1b2c")
        header.pack(fill="x", padx=16, pady=(12, 8))

        tk.Label(
            header,
            text="运行日志",
            fg="#36f7ff",
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(side="left")

        tk.Label(
            header,
            text="实时输出依赖检查、数据库验证、后端与前端启动信息",
            fg="#7a9cab",
            bg="#0a1b2c",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="left", padx=14)

        self.clear_btn = self._button(
            header,
            "清空日志",
            self.clear_logs,
            bg="#10283a",
            fg="#9eefff",
            width=10,
            height=1,
        )
        self.clear_btn.pack(side="right")

        self.log_text = scrolledtext.ScrolledText(
            area,
            bg="#030b12",
            fg="#d9fbff",
            insertbackground="#36f7ff",
            font=("Consolas", 10),
            relief="flat",
            padx=12,
            pady=12,
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.log_text.tag_config("INFO", foreground="#d9fbff")
        self.log_text.tag_config("OK", foreground="#7cffb2")
        self.log_text.tag_config("WARN", foreground="#ffd166")
        self.log_text.tag_config("ERROR", foreground="#ff6b6b")
        self.log_text.tag_config("BACKEND", foreground="#36f7ff")
        self.log_text.tag_config("FRONTEND", foreground="#7cffb2")
        self.log_text.tag_config("SYSTEM", foreground="#ffffff")

    def _build_footer(self):
        footer = tk.Frame(self.main, bg="#06111f")
        footer.pack(fill="x", padx=28, pady=(0, 14))

        tk.Label(
            footer,
            text="© 2026 智慧空气质量预测系统 | 田宇栋",
            fg="#36f7ff",
            bg="#06111f",
            font=("Microsoft YaHei UI", 10),
        ).pack(side="left")

        tk.Label(
            footer,
            text="首次部署建议：依赖检查 → 保存配置 → 验证数据库 → 启动全部",
            fg="#6ea8b5",
            bg="#06111f",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right")

    def _button(self, parent, text, command, bg, fg, width=12, height=2):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            height=height,
            bg=bg,
            fg=fg,
            activebackground="#123b58",
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            cursor="hand2",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        return btn

    def _draw_grid(self):
        self.bg_canvas.delete("grid")

        width = max(self.root.winfo_width(), 1280)
        height = max(self.root.winfo_height(), 840)

        grid_color = "#0b2335"

        for x in range(0, width, 46):
            self.bg_canvas.create_line(x, 0, x, height, fill=grid_color, tags="grid")

        for y in range(0, height, 46):
            self.bg_canvas.create_line(0, y, width, y, fill=grid_color, tags="grid")

        self.bg_canvas.create_line(
            0,
            92,
            width,
            92,
            fill="#0e3d52",
            width=2,
            tags="grid",
        )

        self.root.after(1200, self._draw_grid)

    # =====================================================
    # 日志与状态
    # =====================================================

    def log(self, text, tag="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put((f"[{timestamp}] {text}\n", tag))

    def log_raw(self, text, tag="INFO"):
        self.log_queue.put((text, tag))

    def _poll_log_queue(self):
        try:
            while True:
                text, tag = self.log_queue.get_nowait()
                self.log_text.insert("end", text, tag)
                self.log_text.see("end")
        except queue.Empty:
            pass

        self.root.after(80, self._poll_log_queue)

    def clear_logs(self):
        self.log_text.delete("1.0", "end")

    def update_status_labels(self):
        self.backend_status_var.set(self._status_text(self.backend))
        self.frontend_status_var.set(self._status_text(self.frontend))

        self.backend_pid_var.set(str(self.backend.pid or ("外部" if self.backend.external_running else "--")))
        self.frontend_pid_var.set(str(self.frontend.pid or ("外部" if self.frontend.external_running else "--")))

        backend_running = self.backend.running
        frontend_running = self.frontend.running

        if backend_running and frontend_running:
            self.system_status_var.set("运行中")
        elif backend_running or frontend_running:
            self.system_status_var.set("部分运行")
        elif self.starting:
            self.system_status_var.set("启动中")
        elif self.stopping:
            self.system_status_var.set("停止中")
        else:
            self.system_status_var.set("待启动")

    def _status_text(self, service: ServiceProcess) -> str:
        if service.running:
            if service.external_running and not service.pid:
                return "已运行"

            return "运行中"

        if is_port_open("127.0.0.1", service.port):
            return "端口占用"

        return "未启动"

    def _refresh_status_loop(self):
        self.update_status_labels()
        self.root.after(1000, self._refresh_status_loop)

    # =====================================================
    # 数据库配置
    # =====================================================

    def get_db_config(self) -> dict:
        return {
            "DB_HOST": self.db_host_var.get().strip() or "localhost",
            "DB_PORT": self.db_port_var.get().strip() or "5432",
            "DB_NAME": self.db_name_var.get().strip() or "db1",
            "DB_USER": self.db_user_var.get().strip() or "postgres",
            "DB_PASSWORD": self.db_password_var.get(),
        }

    def save_db_config(self):
        values = self.get_db_config()
        write_env_file(ENV_FILE, values)

        self.log(f"数据库配置已保存：{ENV_FILE}", "OK")
        messagebox.showinfo("保存成功", f"数据库配置已保存到：\n{ENV_FILE}。")

    def build_backend_env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        env.update(self.get_db_config())

        return env

    def validate_database_button(self):
        thread = threading.Thread(target=self._validate_database_worker, daemon=True)
        thread.start()

    def _validate_database_worker(self):
        self.save_db_config()
        ok, detail = self.validate_database_connection()

        if ok:
            self.log("数据库连接验证成功。", "OK")
            messagebox.showinfo("数据库验证成功", detail)
        else:
            self.log(f"数据库连接验证失败：{detail}", "ERROR")
            issues = [
                DependencyIssue(
                    name="PostgreSQL 数据库连接",
                    message=detail,
                    fix=(
                        "1. 确认 PostgreSQL 已安装并正在运行。\n"
                        "2. 确认数据库名、用户名、密码正确。\n"
                        "3. 如果数据库不存在，请先创建数据库，例如 db1。\n"
                        "4. 如果提示 psycopg2 缺失，请执行：python -m pip install psycopg2-binary"
                    ),
                    url=POSTGRESQL_DOWNLOAD_URL,
                    severity="ERROR",
                )
            ]
            self.show_dependency_popup(issues, title="数据库验证失败")

    def validate_database_connection(self):
        try:
            python_cmd = get_python_command()
        except Exception as exc:
            return False, str(exc)

        env = self.build_backend_env()

        check_code = r"""
import os
import sys

try:
    import psycopg2
except Exception as exc:
    print("无法导入 psycopg2：" + str(exc))
    sys.exit(2)

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "db1"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        connect_timeout=5,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.fetchone()
    cur.close()
    conn.close()
    print("PostgreSQL 连接成功")
except Exception as exc:
    print(str(exc))
    sys.exit(3)
"""

        code, output = run_command_capture(
            [python_cmd, "-c", check_code],
            cwd=PROJECT_ROOT,
            env=env,
            timeout=12,
        )

        return code == 0, output

    # =====================================================
    # 依赖检查
    # =====================================================

    def check_dependencies_button(self):
        thread = threading.Thread(target=self._check_dependencies_worker, daemon=True)
        thread.start()

    def _check_dependencies_worker(self):
        self.log("开始依赖检查...", "SYSTEM")
        issues = self.collect_dependency_issues(include_database=True)

        if issues:
            error_count = len([item for item in issues if item.blocking])
            warn_count = len(issues) - error_count

            for issue in issues:
                tag = "ERROR" if issue.blocking else "WARN"
                self.log(f"× {issue.name}：{issue.message}", tag)

            self.log(f"依赖检查完成：{error_count} 个阻断问题，{warn_count} 个提示。", "WARN")
            self.show_dependency_popup(issues, title="依赖检查未通过")
        else:
            self.log("依赖检查通过，当前环境可以启动系统。", "OK")
            messagebox.showinfo("依赖检查通过", "所有关键依赖均已检测通过，可以启动系统。")

    def collect_dependency_issues(self, include_database=False):
        issues = []

        if not (PROJECT_ROOT / "backend").exists():
            issues.append(
                DependencyIssue(
                    "backend 目录",
                    f"未找到 backend 目录：{PROJECT_ROOT / 'backend'}",
                    "请确认启动器位于项目根目录，或者 exe 与 backend、frontend 同级。",
                )
            )

        if not (PROJECT_ROOT / "frontend").exists():
            issues.append(
                DependencyIssue(
                    "frontend 目录",
                    f"未找到 frontend 目录：{PROJECT_ROOT / 'frontend'}",
                    "请确认启动器位于项目根目录，或者 exe 与 backend、frontend 同级。",
                )
            )

        backend_app = PROJECT_ROOT / "backend" / "app.py"
        if not backend_app.exists():
            issues.append(
                DependencyIssue(
                    "backend/app.py",
                    f"未找到后端入口文件：{backend_app}",
                    "请确认 backend/app.py 存在。",
                )
            )

        package_json = FRONTEND_DIR / "package.json"
        if not package_json.exists():
            issues.append(
                DependencyIssue(
                    "frontend/package.json",
                    f"未找到前端依赖文件：{package_json}",
                    "请确认 frontend/package.json 存在。",
                )
            )

        try:
            python_cmd = get_python_command()
            code, output = run_command_capture([python_cmd, "--version"], timeout=10)

            if code == 0:
                self.log(f"√ Python：{output or python_cmd}", "OK")
            else:
                issues.append(
                    DependencyIssue(
                        "Python",
                        output or "Python 无法正常运行",
                        "请安装 Python 3.10+，安装时勾选 Add Python to PATH。",
                        PYTHON_DOWNLOAD_URL,
                    )
                )
                python_cmd = None

        except Exception as exc:
            issues.append(
                DependencyIssue(
                    "Python",
                    str(exc),
                    "请安装 Python 3.10+，安装时勾选 Add Python to PATH。",
                    PYTHON_DOWNLOAD_URL,
                )
            )
            python_cmd = None

        if python_cmd:
            code, output = run_command_capture([python_cmd, "-m", "pip", "--version"], timeout=10)

            if code == 0:
                self.log(f"√ pip：{output}", "OK")
            else:
                issues.append(
                    DependencyIssue(
                        "pip",
                        output or "pip 不可用",
                        "请修复 Python 安装，或执行：python -m ensurepip --upgrade",
                        PYTHON_DOWNLOAD_URL,
                    )
                )

            missing_modules = self.find_missing_python_modules(python_cmd)

            if missing_modules:
                packages = [PYTHON_IMPORT_CHECKS[item] for item in missing_modules]
                install_cmd = self.build_python_install_command(packages)

                issues.append(
                    DependencyIssue(
                        "Python 后端依赖",
                        "缺少模块：" + ", ".join(missing_modules),
                        (
                            "推荐在项目根目录执行：\n"
                            f"{install_cmd}\n\n"
                            "如果仍缺少模块，可执行：\n"
                            f"{PIP_INSTALL_FALLBACK}"
                        ),
                        PYTHON_DOWNLOAD_URL,
                    )
                )
            else:
                self.log("√ Python 后端依赖：关键模块已安装", "OK")

        try:
            npm_cmd = get_npm_command()
            code, output = run_command_capture([npm_cmd, "--version"], timeout=10)

            if code == 0:
                self.log(f"√ npm：{output}", "OK")
            else:
                issues.append(
                    DependencyIssue(
                        "npm",
                        output or "npm 无法正常运行",
                        "请安装 Node.js LTS，安装后重新打开终端或重启电脑。",
                        NODE_DOWNLOAD_URL,
                    )
                )
                npm_cmd = None

        except Exception as exc:
            issues.append(
                DependencyIssue(
                    "Node.js / npm",
                    str(exc),
                    "请安装 Node.js LTS。Node.js 安装包会同时安装 npm。",
                    NODE_DOWNLOAD_URL,
                )
            )
            npm_cmd = None

        if package_json.exists():
            node_modules = FRONTEND_DIR / "node_modules"

            if not node_modules.exists():
                issues.append(
                    DependencyIssue(
                        "前端依赖 node_modules",
                        "未找到 frontend/node_modules，前端依赖尚未安装。",
                        "请在项目根目录执行：\ncd frontend\nnpm install",
                        NODE_DOWNLOAD_URL,
                    )
                )
            else:
                missing_frontend = self.find_missing_frontend_dependencies(package_json, node_modules)
                if missing_frontend:
                    issues.append(
                        DependencyIssue(
                            "前端 package 依赖",
                            "node_modules 中缺少：" + ", ".join(missing_frontend[:20]),
                            "请在 frontend 目录执行：\nnpm install",
                            NODE_DOWNLOAD_URL,
                        )
                    )
                else:
                    self.log("√ 前端依赖：node_modules 已存在，主要依赖目录可找到", "OK")

        if include_database:
            ok, detail = self.validate_database_connection()

            if ok:
                self.log("√ PostgreSQL 数据库连接：通过", "OK")
            else:
                issues.append(
                    DependencyIssue(
                        "PostgreSQL 数据库连接",
                        detail,
                        (
                            "请确认：\n"
                            "1. PostgreSQL 已安装并启动。\n"
                            "2. 启动器中的主机、端口、数据库名、用户名、密码正确。\n"
                            "3. 目标数据库已经创建。\n"
                            "4. 如提示 psycopg2 缺失，请执行：python -m pip install psycopg2-binary"
                        ),
                        POSTGRESQL_DOWNLOAD_URL,
                    )
                )

        return issues

    def build_python_install_command(self, packages):
        req = PROJECT_ROOT / "backend" / "requirements.txt"

        if req.exists():
            return (
                "python -m pip install -r backend\\requirements.txt\n"
                "python -m pip install " + " ".join(sorted(set(packages)))
            )

        return "python -m pip install " + " ".join(sorted(set(packages)))

    def find_missing_python_modules(self, python_cmd):
        missing = []

        for import_name in PYTHON_IMPORT_CHECKS.keys():
            check_code = f"import {import_name}"
            code, _ = run_command_capture(
                [python_cmd, "-c", check_code],
                cwd=PROJECT_ROOT,
                env=self.build_backend_env(),
                timeout=20,
            )

            if code != 0:
                missing.append(import_name)

        return missing

    def find_missing_frontend_dependencies(self, package_json: Path, node_modules: Path):
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception:
            return []

        deps = {}
        deps.update(data.get("dependencies", {}))
        deps.update(data.get("devDependencies", {}))

        missing = []

        for package_name in deps.keys():
            # scoped package: @vitejs/plugin-react -> node_modules/@vitejs/plugin-react
            package_path = node_modules / package_name

            if not package_path.exists():
                missing.append(package_name)

        return missing

    def show_dependency_popup(self, issues, title="依赖检查提示"):
        def show():
            win = tk.Toplevel(self.root)
            win.title(title)
            win.geometry("820x620")
            win.minsize(720, 520)
            win.configure(bg="#06111f")
            win.transient(self.root)

            header = tk.Label(
                win,
                text=title,
                fg="#36f7ff",
                bg="#06111f",
                font=("Microsoft YaHei UI", 18, "bold"),
            )
            header.pack(anchor="w", padx=20, pady=(18, 6))

            summary = tk.Label(
                win,
                text="检测到运行环境尚未就绪。请按照下方提示安装依赖或完成部署后，再重新点击“启动全部”。",
                fg="#b7d6e3",
                bg="#06111f",
                font=("Microsoft YaHei UI", 10),
                wraplength=760,
                justify="left",
            )
            summary.pack(anchor="w", padx=20, pady=(0, 10))

            text = scrolledtext.ScrolledText(
                win,
                bg="#030b12",
                fg="#d9fbff",
                insertbackground="#36f7ff",
                font=("Consolas", 10),
                relief="flat",
                padx=12,
                pady=12,
                wrap="word",
            )
            text.pack(fill="both", expand=True, padx=20, pady=(0, 12))

            content = self.build_dependency_help_text(issues)
            text.insert("1.0", content)
            text.configure(state="disabled")

            buttons = tk.Frame(win, bg="#06111f")
            buttons.pack(fill="x", padx=20, pady=(0, 18))

            urls = []
            seen = set()

            for issue in issues:
                if issue.url and issue.url not in seen:
                    urls.append(issue.url)
                    seen.add(issue.url)

            for url in urls[:4]:
                label = self.url_button_label(url)
                self._button(
                    buttons,
                    label,
                    lambda u=url: webbrowser.open(u),
                    bg="#123b58",
                    fg="#d9fbff",
                    width=16,
                    height=1,
                ).pack(side="left", padx=(0, 8))

            self._button(
                buttons,
                "复制部署提示",
                lambda: self.copy_to_clipboard(content),
                bg="#45326b",
                fg="#efe7ff",
                width=14,
                height=1,
            ).pack(side="right", padx=(8, 0))

            self._button(
                buttons,
                "关闭",
                win.destroy,
                bg="#802020",
                fg="#ffdede",
                width=10,
                height=1,
            ).pack(side="right")

        self.root.after(0, show)

    def url_button_label(self, url):
        if "python.org" in url:
            return "打开 Python"
        if "nodejs.org" in url:
            return "打开 Node.js"
        if "postgresql.org" in url:
            return "打开 PostgreSQL"
        if "git-scm.com" in url:
            return "打开 Git"

        return "打开下载页"

    def build_dependency_help_text(self, issues):
        lines = []
        lines.append("PM2.5 监控预警平台部署检查结果")
        lines.append("=" * 72)
        lines.append("")

        for index, issue in enumerate(issues, start=1):
            mark = "阻断" if issue.blocking else "提示"
            lines.append(f"{index}. [{mark}] {issue.name}")
            lines.append(f"   问题：{issue.message}")
            lines.append("   处理：")
            for fix_line in issue.fix.splitlines():
                lines.append(f"   {fix_line}")
            if issue.url:
                lines.append(f"   下载地址：{issue.url}")
            lines.append("")

        lines.append("=" * 72)
        lines.append("首次部署推荐顺序")
        lines.append("")
        lines.append("1. 安装 Python 3.10+，安装时勾选 Add Python to PATH。")
        lines.append("2. 安装 Node.js LTS，安装后重新打开 PowerShell。")
        lines.append("3. 安装 PostgreSQL，记住 postgres 用户密码，并创建数据库 db1。")
        lines.append("4. 在项目根目录执行：")
        lines.append("   python -m venv .venv1")
        lines.append("   .\\.venv1\\Scripts\\Activate.ps1")
        lines.append("   python -m pip install -r backend\\requirements.txt")
        lines.append("   python -m pip install pandas numpy scikit-learn joblib torch psycopg2-binary python-multipart python-dotenv")
        lines.append("5. 安装前端依赖：")
        lines.append("   cd frontend")
        lines.append("   npm install")
        lines.append("6. 回到启动器，填写数据库信息，点击“保存配置”和“验证数据库”。")
        lines.append("7. 点击“启动全部”。")
        lines.append("")
        lines.append("官方下载地址")
        lines.append(f"Python: {PYTHON_DOWNLOAD_URL}")
        lines.append(f"Node.js: {NODE_DOWNLOAD_URL}")
        lines.append(f"PostgreSQL: {POSTGRESQL_DOWNLOAD_URL}")
        lines.append(f"Git: {GIT_DOWNLOAD_URL}")
        lines.append("")
        lines.append("安全提醒：.env 中包含数据库密码，不要提交到 GitHub。")

        return "\n".join(lines)

    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log("部署提示已复制到剪贴板。", "OK")
        messagebox.showinfo("已复制", "部署提示已复制到剪贴板。")

    # =====================================================
    # 环境检查
    # =====================================================

    def check_environment(self):
        self.log("开始环境检查...", "SYSTEM")

        checks = []

        checks.append(("项目根目录", PROJECT_ROOT.exists(), str(PROJECT_ROOT)))
        checks.append(("backend 目录", (PROJECT_ROOT / "backend").exists(), str(PROJECT_ROOT / "backend")))
        checks.append(("frontend 目录", (PROJECT_ROOT / "frontend").exists(), str(PROJECT_ROOT / "frontend")))
        checks.append(("backend/app.py", (PROJECT_ROOT / "backend" / "app.py").exists(), str(PROJECT_ROOT / "backend" / "app.py")))
        checks.append(("frontend/package.json", (PROJECT_ROOT / "frontend" / "package.json").exists(), str(PROJECT_ROOT / "frontend" / "package.json")))
        checks.append(("frontend/node_modules", (PROJECT_ROOT / "frontend" / "node_modules").exists(), str(PROJECT_ROOT / "frontend" / "node_modules")))

        try:
            python_cmd = get_python_command()
            checks.append(("Python", True, python_cmd))
        except Exception as exc:
            checks.append(("Python", False, str(exc)))

        try:
            npm_cmd = get_npm_command()
            checks.append(("npm", True, npm_cmd))
        except Exception as exc:
            checks.append(("npm", False, str(exc)))

        checks.append((f"后端端口 {BACKEND_PORT}", not is_port_open("127.0.0.1", BACKEND_PORT), "空闲" if not is_port_open("127.0.0.1", BACKEND_PORT) else "已占用"))
        checks.append((f"前端端口 {FRONTEND_PORT}", not is_port_open("127.0.0.1", FRONTEND_PORT), "空闲" if not is_port_open("127.0.0.1", FRONTEND_PORT) else "已占用"))

        ok_count = 0

        for name, ok, detail in checks:
            if ok:
                ok_count += 1
                self.log(f"√ {name}：{detail}", "OK")
            else:
                self.log(f"× {name}：{detail}", "ERROR")

        self.log(f"环境检查完成：{ok_count}/{len(checks)} 项通过", "SYSTEM")

        if ok_count < len(checks):
            issues = self.collect_dependency_issues(include_database=False)
            if issues:
                self.show_dependency_popup(issues, title="环境检查提示")

    # =====================================================
    # 启动逻辑
    # =====================================================

    def start_all(self):
        if self.starting:
            return

        thread = threading.Thread(target=self._start_all_worker, daemon=True)
        thread.start()

    def _start_all_worker(self):
        self.starting = True
        self.update_status_labels()

        self.log("=" * 70, "SYSTEM")
        self.log("开始启动 PM2.5 监控预警平台...", "SYSTEM")
        self.log(f"项目根目录：{PROJECT_ROOT}", "SYSTEM")
        self.log("=" * 70, "SYSTEM")

        try:
            self.save_db_config()

            issues = self.collect_dependency_issues(include_database=True)
            blocking_issues = [item for item in issues if item.blocking]

            if blocking_issues:
                self.log("启动前依赖检查未通过，已中止启动。", "ERROR")
                self.show_dependency_popup(blocking_issues, title="启动前检查未通过")
                return

            self._start_backend()
            self._start_frontend()

            if self.auto_open_browser.get():
                self.log("准备打开浏览器...", "SYSTEM")
                time.sleep(1.2)
                self.open_frontend()

            self.log("启动流程执行完成。", "OK")

        except Exception as exc:
            self.log(f"启动失败：{exc}", "ERROR")
            messagebox.showerror("启动失败", str(exc))

        finally:
            self.starting = False
            self.update_status_labels()

    def _start_backend(self):
        if is_port_open("127.0.0.1", BACKEND_PORT):
            self.backend.external_running = True
            self.log(f"后端端口 {BACKEND_PORT} 已被占用，视为后端已运行：{BACKEND_URL}", "WARN")
            return

        python_cmd = get_python_command()

        env = self.build_backend_env()

        cmd = [
            python_cmd,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--reload",
            "--port",
            str(BACKEND_PORT),
        ]

        self.log("正在启动后端 FastAPI...", "BACKEND")
        self.log("后端命令：" + " ".join(cmd), "BACKEND")

        creation_flags = 0

        if is_windows():
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )

        self.backend.process = process
        self.backend.external_running = False

        threading.Thread(
            target=self._stream_output,
            args=(process, "后端", "BACKEND"),
            daemon=True,
        ).start()

        self.log("等待后端端口启动...", "BACKEND")

        if wait_for_port("127.0.0.1", BACKEND_PORT, timeout=40):
            self.log(f"后端启动成功：{BACKEND_URL}", "OK")
        else:
            self.log("后端启动等待超时，请查看日志。", "ERROR")

    def _start_frontend(self):
        if is_port_open("127.0.0.1", FRONTEND_PORT):
            self.frontend.external_running = True
            self.log(f"前端端口 {FRONTEND_PORT} 已被占用，视为前端已运行：{FRONTEND_URL}", "WARN")
            return

        package_json = FRONTEND_DIR / "package.json"

        if not package_json.exists():
            raise RuntimeError(f"未找到 frontend/package.json：{package_json}")

        node_modules = FRONTEND_DIR / "node_modules"

        if not node_modules.exists():
            raise RuntimeError(
                "未找到 frontend/node_modules。\n\n"
                "请先在 frontend 目录执行：\n"
                "npm install"
            )

        npm_cmd = get_npm_command()

        cmd = [
            npm_cmd,
            "run",
            "dev",
            "--",
            "--host",
            "0.0.0.0",
        ]

        self.log("正在启动前端 Vite...", "FRONTEND")
        self.log("前端命令：" + " ".join(cmd), "FRONTEND")

        creation_flags = 0

        if is_windows():
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(
            cmd,
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )

        self.frontend.process = process
        self.frontend.external_running = False

        threading.Thread(
            target=self._stream_output,
            args=(process, "前端", "FRONTEND"),
            daemon=True,
        ).start()

        self.log("等待前端端口启动...", "FRONTEND")

        if wait_for_port("127.0.0.1", FRONTEND_PORT, timeout=40):
            self.log(f"前端启动成功：{FRONTEND_URL}", "OK")
        else:
            self.log("前端启动等待超时，请查看日志。", "ERROR")

    def _stream_output(self, process, prefix, tag):
        if not process.stdout:
            return

        for line in iter(process.stdout.readline, ""):
            if not line:
                break

            self.log_raw(f"[{prefix}] {line}", tag)

        code = process.poll()

        if code is not None:
            self.log(f"{prefix}进程已退出，退出码：{code}", "WARN")

    # =====================================================
    # 停止逻辑
    # =====================================================

    def stop_all(self):
        if self.stopping:
            return

        thread = threading.Thread(target=self._stop_all_worker, daemon=True)
        thread.start()

    def _stop_all_worker(self):
        self.stopping = True
        self.update_status_labels()

        self.log("正在停止所有服务...", "SYSTEM")

        self._stop_service(self.frontend, "前端")
        self._stop_service(self.backend, "后端")

        self.log("停止流程执行完成。", "OK")

        self.stopping = False
        self.update_status_labels()

    def _stop_service(self, service: ServiceProcess, label: str):
        if service.process and service.process.poll() is None:
            pid = service.process.pid
            self.log(f"正在停止{label}进程 PID={pid}", "SYSTEM")

            try:
                if is_windows():
                    taskkill_pid(pid)
                else:
                    service.process.terminate()
                    time.sleep(1)

                    if service.process.poll() is None:
                        service.process.kill()

                self.log(f"{label}进程已停止。", "OK")

            except Exception as exc:
                self.log(f"{label}停止失败：{exc}", "ERROR")

        elif service.external_running:
            self.log(f"{label}是外部已运行服务，启动器不会强制停止。", "WARN")

        else:
            self.log(f"{label}未启动，无需停止。", "INFO")

        service.process = None
        service.external_running = False

    def restart_all(self):
        def worker():
            self._stop_all_worker()
            time.sleep(1.5)
            self._start_all_worker()

        threading.Thread(target=worker, daemon=True).start()

    # =====================================================
    # 其他
    # =====================================================

    def open_frontend(self):
        if is_port_open("127.0.0.1", FRONTEND_PORT):
            webbrowser.open(FRONTEND_URL)
            self.log(f"已打开系统页面：{FRONTEND_URL}", "OK")
        else:
            self.log("前端服务尚未启动，无法打开页面。", "WARN")
            messagebox.showwarning("前端未启动", "前端服务尚未启动，无法打开页面。")

    def on_close(self):
        backend_running = self.backend.running
        frontend_running = self.frontend.running

        if backend_running or frontend_running:
            ok = messagebox.askyesno(
                "确认退出",
                "检测到前端或后端服务仍在运行。\n\n"
                "是否停止服务并退出启动器？"
            )

            if ok:
                self._stop_all_worker()
                self.root.destroy()
            else:
                self.root.destroy()
        else:
            self.root.destroy()


def get_resource_path(filename: str) -> Path:
    """
    兼容普通 Python 运行和 PyInstaller 打包运行。
    """
    candidates = []

    if getattr(sys, "frozen", False):
        # PyInstaller --onefile 解压后的临时目录
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / filename)
        # exe 所在目录
        candidates.append(Path(sys.executable).resolve().parent / filename)

    # 项目根目录
    candidates.append(PROJECT_ROOT / filename)

    # start_all.py 所在目录
    try:
        candidates.append(Path(__file__).resolve().parent / filename)
    except Exception:
        pass

    for path in candidates:
        if path and path.exists():
            return path

    return PROJECT_ROOT / filename


def set_windows_app_id():
    """
    让 Windows 任务栏使用当前程序自己的图标，而不是默认 Tk 图标。
    """
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "TianYuDong.PM25.WarningPlatform.Launcher"
            )
        except Exception:
            pass


def apply_window_icon(root):
    icon_path = get_resource_path("icon_fox.ico")

    if icon_path.exists():
        try:
            root.iconbitmap(default=str(icon_path))
        except Exception:
            pass


def main():
    set_windows_app_id()

    root = tk.Tk()
    apply_window_icon(root)

    try:
        app = TechLauncher(root)
        apply_window_icon(root)

        app.log("启动器已就绪。", "OK")
        app.log("首次部署建议：点击“依赖检查”，根据弹窗安装缺失依赖。", "SYSTEM")
        app.log("依赖就绪后：填写数据库配置 → 保存配置 → 验证数据库 → 启动全部。", "SYSTEM")

        root.mainloop()

    except Exception as exc:
        messagebox.showerror("启动器初始化失败", str(exc))
        raise


if __name__ == "__main__":
    main()
