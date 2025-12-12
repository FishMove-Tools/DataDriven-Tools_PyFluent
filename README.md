![image](https://github.com/FishMove-Tools/DataDriven-Tools_PyFluent/blob/main/CFD%20train%20stage/Two_stage_training_pipeline.png)

# FishMove Tools

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=21&pause=500&color=677BF7&center=%E5%81%87&vCenter=%E5%81%87&multiline=true&repeat=%E7%9C%9F&random=%E5%81%87&width=480&height=60&lines=Welcome+to+FishMove+Toolkit!;%E6%AC%A2%E8%BF%8E%E4%BD%BF%E7%94%A8FishMove%E7%B3%BB%E5%88%97%E5%B7%A5%E5%85%B7)](https://git.io/typing-svg)
### 🧠 Tech Stack / Tags

![](https://img.shields.io/badge/DeepRL-%23369FF7FF)  ![](https://img.shields.io/badge/BioRobotics-%23669FF7FF)  ![](https://img.shields.io/badge/Control-%23766BF7FF)  ![](https://img.shields.io/badge/FluidSimulation-%23766BF7FF)  ![](https://img.shields.io/badge/FishModeling-%23669FF7FF)  ![](https://img.shields.io/badge/GymEnv-%2366BB66FF)

---

### 📦 Featured Projects
- 🐠 `FishdatadrivenEnv`: A gym environment for reinforcement learning of fish swimming dynamics.
- 📊 `FishDynamicsModel`: A PyTorch-based data-driven prediction model for soft-body robotic fish.
- 🧪 `RL-FishControl`: A reinforcement learning controller for underwater navigation.

> 💬 Contributions & discussions welcome! Feel free to check out our tools, raise issues, or star the projects.

## 📋 Contents
- [🏠 About](#-about)
- [📚 Getting Started](#-getting-started)
- [📦 Benchmark & Method](#-benchmark--method)
- [👥 Support](#-support)
- [📝 TODO List](#-todo-list)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

---

## 🏠 About

FishMove Toolkit is a general-purpose toolkit focused on **bio-inspired underwater robot control** and **data-driven fluid dynamics** research.

Given the prohibitive computational cost of Computational Fluid Dynamics (CFD) simulation for training, we adopt a **data-driven Sim-to-CFD** paradigm, combining data-driven models and Reinforcement Learning (RL) to achieve efficient control strategy learning for underwater robots.

Key features of this toolkit include:

* **⚡ Efficient RL Training Environment:** Provides `FishdatadrivenEnv` (Gym-based) that uses a data-driven model to replace time-consuming CFD, significantly accelerating the RL pre-training process (PPO).
* **🐟 General Control Framework:** Enables training robotic fish agents for navigation and manipulation in diverse fluid environments (e.g., turbulent or quiescent water).
* **🚀 Sim-to-CFD Strategy:** Supports transferring pre-trained policies from the fast simulation environment into the ANSYS Fluent (CFD) environment via the `pyfluent` interface for **policy fine-tuning**, ensuring more accurate Sim-to-Real migration.

---

## 📚 Getting Started

### Prerequisites

* Operating System: Windows or Linux (Ubuntu 20.04+ recommended)
* NVIDIA GPU (Optional, but recommended for PyTorch training)
* **ANSYS Fluent** (Must be installed and configured for `ansys-fluent-core`)
* Conda
* Python 3.9

### Installation

The installation is divided into two main parts: Python Environment Setup and Pyfluent Setup.

#### 1. Python Environment Setup

It is recommended to use Conda to create an isolated environment for installation.

1.  **Create and Activate Conda Environment:**

    ```bash
    conda create -n fish python=3.9.13
    conda activate fish
    ```

2.  **Install Main Dependencies:**

    ```bash
    # Upgrade pip
    pip install --upgrade pip

    # Install core libraries (Deep Learning and RL)
    pip install numpy==2.0.2
    pip install torch==2.1.0
    pip install stable-baselines3[extra]
    
    # Install Pyfluent package
    pip install ansys-fluent-core
    
    # Adjust Pandas version (to prevent conflicts)
    pip uninstall pandas -y
    pip install pandas==2.2.2
    ```

3.  **Clone the Project Repository:**

    ```bash
    git clone [https://github.com/Zhan-Sun/FishMoveTools.git](https://github.com/Zhan-Sun/FishMoveTools.git)
    cd FishMoveTools
    
    # Install FishMove Tools package itself (if applicable)
    # pip install -e . 
    ```

#### 2. Pyfluent Setup

The `ansys-fluent-core` package allows seamless control of ANSYS Fluent from Python.

**A. Pyfluent Installation and Documentation**

* The Python package for Pyfluent is installed via `pip install ansys-fluent-core` (completed in the step above).
* For detailed documentation and installation guides regarding ANSYS Fluent and the Pyfluent package, refer to the official repository: [https://github.com/leigq/pyfluent](https://github.com/leigq/pyfluent).

**B. Controlling Fluent via Python (Jupyter)**

Pyfluent commands are executed through a Python session (e.g., in a Jupyter Notebook) to control the CFD simulation:

| Operation | Command | Description |
| :--- | :--- | :--- |
| **Import** | `import ansys.fluent.core as pyfluent` | Imports the core library. |
| **Launch (No GUI)** | `session = pyfluent.launch_fluent()` | Starts Fluent without the graphical user interface. |
| **Launch (With GUI)** | `session = pyfluent.launch_fluent(show_gui = True)` | Starts Fluent with the GUI enabled (only available in meshing mode). |
| **Exit** | `session.exit()` | Closes the Fluent session. |

**C. Interaction Modes**

* **Meshing Mode:** Supports GUI interaction.
* **Solution Mode:** Only supports Text User Interface (TUI) interaction.
* **Loading Mesh Example (TUI):** `session.tui.file.read_case("mesh fish .msh")`

**D. Pyfluent Journaling (Code Generation)**

Pyfluent supports journaling to automatically generate Python code from TUI commands:

1.  Launch Fluent (e.g., `session = pyfluent.launch_fluent()`).
2.  In the Fluent TUI console, start recording the journal: `(api-start-python-journal "python_journal.py")`
3.  Perform parameter settings and commands via the TUI. The corresponding Python code will be written to `python_journal.py` in the working directory.
4.  Stop recording: `(api-stop-python-journal)`
5.  This generated Python code can be modified and used directly within your Pyfluent scripts in Jupyter for configuration.

**E. Command APIs**

Pyfluent offers two ways to interact with Fluent:
* **TUI API:** Mimics the traditional TUI console commands (e.g., `session.tui.file.read_case`).
* **Settings API:** Based on a hierarchical (tree-like) structure for settings (still under development).

---


## 📝 TODO List
- \[x\] Release Release policy fine-tuning part with pyfluent interface.
- \[ \] Release pre-training code.
- \[ \] Release the PD-FS framework.
- \[ \] Release the paper with demos.
