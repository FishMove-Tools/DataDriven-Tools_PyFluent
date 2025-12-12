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
    
    # Install Ansys Fluent Python interface
    pip install ansys-fluent-core
    
    # Adjust Pandas version (to prevent conflicts)
    pip uninstall pandas -y
    pip install pandas==2.2.2
    ```
    > ⚠️ **Note:** Installation of `ansys-fluent-core` requires your local **ANSYS Fluent** environment to be correctly configured.

3.  **Clone the Project Repository:**

    ```bash
    git clone [https://github.com/Zhan-Sun/FishMoveTools.git](https://github.com/Zhan-Sun/FishMoveTools.git)
    cd FishMoveTools
    
    # Install FishMove Tools package itself (if applicable)
    # pip install -e . 
    ```

---


## 📝 TODO List
- \[x\] Release Release policy fine-tuning part with pyfluent interface.
- \[ \] Release pre-training code.
- \[ \] Release the PD-FS framework.
- \[ \] Release the paper with demos.
