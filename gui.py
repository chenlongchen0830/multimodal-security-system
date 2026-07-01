# -*- coding: utf-8 -*-
"""
智能安防监控系统 — GUI 可视化界面（Tkinter）

用法：
    python gui.py

功能：
    - 图形化选择视频、功能、参数
    - 一键运行 main_v2.py
    - 实时日志输出
    - 无需记忆命令行参数
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import sys
import subprocess
import threading
import glob


class SmartSecurityGUI:
    """智能安防监控系统 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("智能安防监控系统 — 可视化启动器")
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        
        # 检查项目目录
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.project_dir)
        
        self._build_ui()
        self._scan_videos()
    
    def _build_ui(self):
        """构建界面"""
        # ===== 顶部标题 =====
        title = tk.Label(self.root, text="🔒 智能安防监控系统", font=("微软雅黑", 18, "bold"))
        title.pack(pady=10)
        
        subtitle = tk.Label(self.root, text="基于计算机视觉的入侵检测 / 车辆识别 / 人流统计 / 人脸识别", font=("微软雅黑", 10))
        subtitle.pack()
        
        # ===== 主框架 =====
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # ----- 左侧：配置面板 -----
        left_frame = tk.LabelFrame(main_frame, text="⚙️ 运行配置", font=("微软雅黑", 11), padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 视频源
        tk.Label(left_frame, text="视频源：", font=("微软雅黑", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(left_frame, textvariable=self.video_var, width=35, state="readonly")
        self.video_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 摄像头选项
        self.camera_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left_frame, text="使用摄像头 (0)", variable=self.camera_var, font=("微软雅黑", 9),
                       command=self._toggle_camera).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # 分隔线
        tk.Frame(left_frame, height=2, bg="#ccc").grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        
        # 功能选择
        tk.Label(left_frame, text="功能选择：", font=("微软雅黑", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=5)
        
        self.all_var = tk.BooleanVar(value=True)
        self.face_var = tk.BooleanVar(value=False)
        self.blacklist_var = tk.BooleanVar(value=False)
        self.intrusion_var = tk.BooleanVar(value=False)
        self.vehicle_var = tk.BooleanVar(value=False)
        self.flow_var = tk.BooleanVar(value=False)
        
        tk.Checkbutton(left_frame, text="🚀 全功能模式 (--all)", variable=self.all_var, font=("微软雅黑", 9, "bold"),
                       command=self._toggle_all).grid(row=3, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="👤 人脸识别 (--face)", variable=self.face_var, font=("微软雅黑", 9)).grid(row=4, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="⛔ 黑名单报警 (--blacklist)", variable=self.blacklist_var, font=("微软雅黑", 9)).grid(row=5, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="🚫 入侵检测 (--intrusion)", variable=self.intrusion_var, font=("微软雅黑", 9)).grid(row=6, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="🚗 车辆识别 (--vehicle)", variable=self.vehicle_var, font=("微软雅黑", 9)).grid(row=7, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="🚶 人流统计 (--flow)", variable=self.flow_var, font=("微软雅黑", 9)).grid(row=8, column=1, sticky=tk.W, pady=2)
        
        # 分隔线
        tk.Frame(left_frame, height=2, bg="#ccc").grid(row=9, column=0, columnspan=2, sticky="ew", pady=8)
        
        # 选项
        tk.Label(left_frame, text="选项：", font=("微软雅黑", 10, "bold")).grid(row=10, column=0, sticky=tk.W, pady=5)
        
        self.interactive_var = tk.BooleanVar(value=False)
        self.reset_zone_var = tk.BooleanVar(value=False)
        self.no_show_var = tk.BooleanVar(value=False)
        
        tk.Checkbutton(left_frame, text="🖱️ 交互式框选禁区 (--interactive)", variable=self.interactive_var, font=("微软雅黑", 9)).grid(row=10, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="🔄 强制重新框禁区 (--reset-zone)", variable=self.reset_zone_var, font=("微软雅黑", 9)).grid(row=11, column=1, sticky=tk.W, pady=2)
        tk.Checkbutton(left_frame, text="🔕 后台处理（不弹窗）(--no-show)", variable=self.no_show_var, font=("微软雅黑", 9)).grid(row=12, column=1, sticky=tk.W, pady=2)
        
        # 底部提示
        tk.Label(left_frame, text="提示：全功能模式已包含入侵+车辆+人流，可选加人脸",
                 font=("微软雅黑", 8), fg="gray").grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # ----- 右侧：按钮 + 日志 -----
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # 运行按钮
        self.run_btn = tk.Button(right_frame, text="▶ 运行系统", font=("微软雅黑", 14, "bold"),
                                 bg="#4CAF50", fg="white", width=15, height=2, command=self._run_system)
        self.run_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(right_frame, text="⏹ 停止运行", font=("微软雅黑", 12),
                                  bg="#f44336", fg="white", width=15, command=self._stop_system)
        self.stop_btn.pack(pady=5)
        
        # 日志区域
        log_label = tk.Label(right_frame, text="📋 运行日志：", font=("微软雅黑", 10, "bold"))
        log_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.log_area = scrolledtext.ScrolledText(right_frame, width=40, height=20, font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_area.insert(tk.END, "系统就绪，请选择配置后点击 [运行系统]\n")
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, font=("微软雅黑", 9), bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _scan_videos(self):
        """扫描视频文件"""
        video_dir = os.path.join(self.project_dir, "videos")
        videos = []
        if os.path.exists(video_dir):
            for ext in ("*.mp4", "*.avi", "*.mov"):
                videos.extend(glob.glob(os.path.join(video_dir, ext)))
            videos = [os.path.basename(v) for v in videos]
        
        if videos:
            self.video_combo["values"] = videos
            self.video_combo.current(0)
        else:
            self.video_combo["values"] = ["（无视频文件）"]
            self.video_combo.current(0)
    
    def _toggle_camera(self):
        """切换摄像头模式"""
        if self.camera_var.get():
            self.video_combo.set("摄像头 (0)")
            self.video_combo.config(state="disabled")
        else:
            self.video_combo.config(state="readonly")
            self._scan_videos()
    
    def _toggle_all(self):
        """全功能开关联动"""
        if self.all_var.get():
            # 全功能开启时，单独功能可选但非必须
            pass
    
    def _build_command(self):
        """构建命令行参数"""
        cmd = [sys.executable, "main_v2.py"]
        
        # 视频源
        if self.camera_var.get():
            cmd.extend(["--source", "0"])
        else:
            video = self.video_var.get()
            if video and video != "（无视频文件）":
                cmd.extend(["--source", f"videos/{video}"])
            else:
                messagebox.showwarning("警告", "请先选择视频文件！")
                return None
        
        # 功能
        if self.all_var.get():
            cmd.append("--all")
        if self.face_var.get():
            cmd.append("--face")
        if self.blacklist_var.get():
            cmd.append("--blacklist")
        if self.intrusion_var.get():
            cmd.append("--intrusion")
        if self.vehicle_var.get():
            cmd.append("--vehicle")
        if self.flow_var.get():
            cmd.append("--flow")
        
        # 选项
        if self.interactive_var.get():
            cmd.append("--interactive")
        if self.reset_zone_var.get():
            cmd.append("--reset-zone")
        if self.no_show_var.get():
            cmd.append("--no-show")
        
        return cmd
    
    def _run_system(self):
        """运行系统"""
        cmd = self._build_command()
        if cmd is None:
            return
        
        self.log_area.insert(tk.END, f"\n{'='*40}\n")
        self.log_area.insert(tk.END, f"执行命令: {' '.join(cmd)}\n")
        self.log_area.insert(tk.END, f"{'='*40}\n")
        self.log_area.see(tk.END)
        
        self.status_var.set("运行中...")
        self.run_btn.config(state="disabled", text="运行中...")
        
        # 在新线程运行，避免阻塞 GUI
        threading.Thread(target=self._execute, args=(cmd,), daemon=True).start()
    
    def _execute(self, cmd):
        """执行命令"""
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in self.process.stdout:
                self.root.after(0, self._append_log, line)
            
            self.process.wait()
            returncode = self.process.returncode
            
            if returncode == 0:
                self.root.after(0, self._append_log, "\n[✅] 运行完成！\n")
                self.root.after(0, self.status_var.set, "运行完成")
            else:
                self.root.after(0, self._append_log, f"\n[❌] 运行异常，退出码: {returncode}\n")
                self.root.after(0, self.status_var.set, "运行异常")
                
        except Exception as e:
            self.root.after(0, self._append_log, f"\n[❌] 执行错误: {e}\n")
            self.root.after(0, self.status_var.set, "执行错误")
        
        finally:
            self.root.after(0, self.run_btn.config, {"state": "normal", "text": "▶ 运行系统"})
    
    def _append_log(self, text):
        """追加日志"""
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)
    
    def _stop_system(self):
        """停止运行"""
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()
            self._append_log("\n[⏹] 已发送停止信号\n")
            self.status_var.set("已停止")
        else:
            self._append_log("\n[⚠️] 没有正在运行的进程\n")
    
    def on_closing(self):
        """关闭窗口"""
        self._stop_system()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SmartSecurityGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
