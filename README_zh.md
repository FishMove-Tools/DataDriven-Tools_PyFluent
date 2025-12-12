<div>
  <h1>
    🐠 FishMove Tools&nbsp;&nbsp;&nbsp;
    <span style="float: right; font-size: 16px; font-weight: normal; margin-top: 10px;">
      <a href="README.md"> 🌎 English </a> | <b>中文</b>
    </span>
  </h1>
</div>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=21&pause=500&color=677BF7&center=%E5%81%87&vCenter=%E5%81%87&multiline=true&repeat=%E7%9C%9F&random=%E5%81%87&width=480&height=60&lines=Welcome+to+FishMove+Toolkit!;%E6%AC%A2%E8%BF%8E%E4%BD%BF%E7%94%A8FishMove%E7%B3%BB%E5%88%97%E5%B7%A5%E5%85%B7)](https://git.io/typing-svg)

<p align="center">
  <img src="https://github.com/FishMove-Tools/DataDriven-Tools_PyFluent/blob/main/CFD%20train%20stage/Two_stage_training_pipeline.png?raw=true" width="100%">
</p>

---

### 🧠 技术栈 / 标签

![](https://img.shields.io/badge/DeepRL-%23369FF7FF)  ![](https://img.shields.io/badge/BioRobotics-%23669FF7FF)  ![](https://img.shields.io/badge/Control-%23766BF7FF)  ![](https://img.shields.io/badge/FluidSimulation-%23766BF7FF)  ![](https://img.shields.io/badge/FishModeling-%23669FF7FF)  ![](https://img.shields.io/badge/GymEnv-%2366BB66FF)

---

### 📦 创新点
- 🐠 `FishdatadrivenEnv`: 用于鱼游动力学的强化学习仿真环境。
- 📊 `FishDynamicsModel`: 鱼形机器人数据驱动预测动力学模型。
- 🧪 `RL-FishControl`: 用于水下导航的强化学习控制器。

> 💬 欢迎贡献与讨论！欢迎查看我们的工具，提出 Issue 或为项目点赞 Star。

## 📋 目录
- [🏠 关于项目](#-关于项目)
- [📚 开始使用](#-开始使用)
- [📦 基准测试与方法](#-基准测试与方法)
- [👥 支持](#-支持)
- [📝 待办事项](#-待办事项)
- [🔗 引用](#-引用)
- [📄 许可证](#-许可证)
- [👏 致谢](#-致谢)

---

## 🏠 关于项目

FishMove Toolkit 是一个专注于**仿生水下机器人控制**和**数据驱动流体力学**研究的通用工具箱。

由于计算流体力学 (CFD) 模拟的训练成本极高，我们采用了**数据驱动的 Sim-to-CFD** 范式，结合数据驱动模型和强化学习 (RL)，实现高效的水下机器人控制策略学习。


**核心特性：**
* **⚡ 高效 RL 训练环境**：提供基于 Gym 的 `FishdatadrivenEnv`，使用数据驱动模型替代耗时的 CFD，大幅提升 RL 预训练速度 (PPO)，并解决没有IB-LBM情况下的动网格仿真问题。
* **🐟 通用控制框架**：支持在多种流体环境（如湍流或静水）中训练机器人鱼的导航与操纵策略。
* **🚀 Sim-to-CFD 策略**：支持通过 `pyfluent` 接口将快速仿真环境中的预训练策略迁移至 ANSYS Fluent (CFD) 环境进行**策略微调**，确保更精确的 Sim-to-Real 迁移。

---

## 📚 开始使用

### 环境要求
* 操作系统: Windows 或 Linux (推荐 Ubuntu 20.04+)
* GPU: NVIDIA GPU (推荐用于 PyTorch 训练)
* **ANSYS Fluent**: 必须安装并配置好环境以支持 `ansys-fluent-core`
* Python 环境: Conda (Python 3.9)

### 安装步骤

安装分为两部分：Python 环境配置与 Pyfluent 配置。

#### 1. Python 环境配置

1.  **创建并激活 Conda 环境：**
    ```bash
    conda create -n fish python=3.9.13
    conda activate fish
    ```

2.  **安装核心依赖：**
    ```bash
    # 升级 pip
    pip install --upgrade pip

    # 安装核心库 (深度学习与 RL)
    pip install numpy==2.0.2
    pip install torch==2.1.0
    pip install stable-baselines3[extra]
    
    # 安装 Pyfluent 包
    pip install ansys-fluent-core
    
    # 调整 Pandas 版本 (避免冲突)
    pip uninstall pandas -y
    pip install pandas==2.2.2
    ```

3.  **克隆项目仓库：**
    ```bash
    git clone [https://github.com/Zhan-Sun/FishMoveTools.git](https://github.com/Zhan-Sun/FishMoveTools.git)
    cd FishMoveTools
    ```

#### 2. Pyfluent 配置

`ansys-fluent-core` 允许通过 Python 无缝控制 ANSYS Fluent。

* **文档参考**: 请参考官方仓库 [pyfluent](https://github.com/leigq/pyfluent)。
* **控制命令示例 (Jupyter)**:

| 操作 | 命令 | 说明 |
| :--- | :--- | :--- |
| **导入** | `import ansys.fluent.core as pyfluent` | 导入核心库 |
| **启动 (无 GUI)** | `session = pyfluent.launch_fluent()` | 在后台启动 Fluent |
| **启动 (带 GUI)** | `session = pyfluent.launch_fluent(show_gui=True)` | 启动带界面的 Fluent (仅限 meshing 模式) |
| **退出** | `session.exit()` | 关闭会话 |

* **录制脚本 (Journaling)**:
  1. 在 Fluent TUI 中输入：`(api-start-python-journal "python_journal.py")` 开始录制。
  2. 执行操作后，输入：`(api-stop-python-journal)` 停止录制。
  3. 生成的 `.py` 文件可直接在 Jupyter 中运行。

---

## 📝 待办事项
- [x] 发布基于 pyfluent 接口的 CFD 训练微调代码。
- [ ] 发布第一阶段预训练代码。
- [ ] 发布第二阶段预训练代码。
- [ ] 发布 PD-FS 框架。
- [ ] 发布带有演示视频的论文。

---

