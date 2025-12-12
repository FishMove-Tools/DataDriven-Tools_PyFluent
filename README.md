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

## Modification log
The toolkit was created by Zhan Ruixin, Georgia Institute of Technology and Sun Weiyuan, TsingHua University.  
This toolkit can train the control of robot fish agent in different flow environments, with less resource consumption compared with CFD.  
We plan to perform pre-training in a fast simulation environment（Gym and train by PPO） and plug the obtained strategies into CFD for further training to reduce the training time.  
Undoubtedly, the results are remarkable！  


### 2025/3/25 修改日志  
在前期的修改中，我们已经完成了工具的基本框架，详情可见3.26汇报PPT。  

其中，有一些值得关注的技术细节：  
  1.在PPO与CFD的EVN中，我们对动作进行了裁剪，即限制每次动作的变化不能超过1从而防止CFD环境中负网格的出现。  
  2.我们在快速仿真环境中选择的动作步数是300步，而CFD中为3000步。这意味着切换到CFD后每个动作将重复10次，最后达到的时间为15s。  
  
同时，有一些待解决的问题也值得考究：  
  1.我们已经发现动作的变化会导致涡结构异常，使推进力大幅下降，我们判定这是从数据驱动切换至CFD产生的GAP。主要问题有两点：切换动作并非在平衡位置导致涡位置不在水平线；不同动作的涡相互影响。
