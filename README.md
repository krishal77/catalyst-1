# 🚦 Catalyst11: Autonomous SUMO Traffic Signal Control via Reinforcement Learning

🥈 **2nd Place Winner** — Provincial Level Hackathon

**Catalyst11** is an Intelligent Transportation System (ITS) platform utilizing **Eclipse SUMO (Simulation of Urban MObility)** and **TraCI (Traffic Control Interface)** to dynamically optimize traffic signal phase timings at complex urban intersections using **Deep Reinforcement Learning (DQN)**.

---

## 🧠 Technical Architecture & Reinforcement Learning Formulation

Catalyst11 models urban intersection signal control as a continuous-space **Markov Decision Process (MDP)** defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

```
                    +--------------------------------+
                    |     Eclipse SUMO Environment   |
                    | (Microscopic Traffic Simulator)|
                    +---------------+----------------+
                                    |
            E2 Induction Loop       | TraCI API
            Detector Telemetry      | (TCP/IPC Socket)
                                    v
                    +---------------+----------------+
                    |      State Space Extraction    |
                    |  7-D Feature Vector S_t        |
                    +---------------+----------------+
                                    |
                                    v
                    +---------------+----------------+
                    |       Deep Q-Network Agent     |
                    |      (Keras / TensorFlow)      |
                    +---------------+----------------+
                                    |
            Action Choice a_t       | Phase Transition &
            (Keep vs Switch)        | Min Green Guard
                                    v
                    +---------------+----------------+
                    |   Traffic Signal Controller    |
                    |        (Junction Node2)        |
                    +--------------------------------+
```

### 1. State Space ($\mathcal{S}$)
The state vector $S_t \in \mathbb{R}^7$ captures real-time microscopic traffic density obtained directly from six **E2 Lane-Area Induction Loop Detectors** positioned at critical intersection approach lanes, alongside the active traffic signal phase index:

$$S_t = \Big( q_{\text{EB},0},\; q_{\text{EB},1},\; q_{\text{EB},2},\; q_{\text{SB},0},\; q_{\text{SB},1},\; q_{\text{SB},2},\; \phi_t \Big)$$

- $q_{\text{EB},0..2}$: Vehicle queue lengths on Eastbound approach lanes (`Node1_2_EB_0`, `Node1_2_EB_1`, `Node1_2_EB_2`).
- $q_{\text{SB},0..2}$: Vehicle queue lengths on Southbound approach lanes (`Node2_7_SB_0`, `Node2_7_SB_1`, `Node2_7_SB_2`).
- $\phi_t \in \{0, 1, 2, 3\}$: Active traffic light phase index at junction `Node2`.

### 2. Action Space ($\mathcal{A}$)
The agent selects a discrete control decision $a_t \in \{0, 1\}$ at each decision epoch:
- **$a_t = 0$ (Maintain Phase)**: Keep current green light signal phase active.
- **$a_t = 1$ (Phase Transition)**: Initiate a green light phase switch to the next valid signal configuration.

### 3. Safety Constraints & Min Green Guard
To prevent phase rapid-cycling, dynamic signal flickering, and ensure vehicle intersection clearance safety, an imperative **Minimum Green Time Guard** is enforced:

$$\Delta t_{\text{switch}} = t_{\text{current}} - t_{\text{last\_switch}} \ge \text{MIN\_GREEN\_STEPS} \quad (\text{MIN\_GREEN\_STEPS} = 100 \text{ steps})$$

Phase switching ($a_t = 1$) is ignored until $\Delta t_{\text{switch}} \ge \text{MIN\_GREEN\_STEPS}$, maintaining operational traffic safety.

### 4. Reward Function ($\mathcal{R}$)
The objective function incentivizes latency minimization and queue length reduction. The instantaneous reward $R_t$ is computed as the negative aggregate queue count across all monitored approach lanes:

$$R(S_t, a_t) = - \sum_{i=1}^{6} q_i = - \Big( q_{\text{EB},0} + q_{\text{EB},1} + q_{\text{EB},2} + q_{\text{SB},0} + q_{\text{SB},1} + q_{\text{SB},2} \Big)$$

Maximizing cumulative discounted return $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ directly minimizes traffic congestion and idling times.

---

## 💻 Deep Q-Network Mechanics (`agents/agent.py`)

The DQN agent leverages a Deep Neural Network built using **Keras/TensorFlow** to approximate the optimal action-value function $Q^*(s, a) \approx Q(s, a; \theta)$:

### Neural Network Architecture
```
Input Layer (7) ---> Dense (24 units, ReLU) ---> Dense (24 units, ReLU) ---> Output Layer (2 units, Linear)
```

- **Loss Function**: Mean Squared Error ($\text{MSE}$) between target and predicted Q-values.
- **Optimizer**: Adam ($\eta = 0.001$).
- **Bellman Target Update**:

$$y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)$$

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \Big( y_t - Q(S_t, a_t; \theta) \Big)^2 \right]$$

### Hyperparameters
| Parameter | Description | Value |
| :--- | :--- | :--- |
| `maxSteps` | Total simulation execution steps | `10000` |
| `learnRate` ($\alpha$) | Learning rate multiplier | `0.1` |
| `discount` ($\gamma$) | Discount factor for future rewards | `0.9` |
| `randChance` ($\varepsilon$) | Epsilon-greedy exploration probability | `0.1` |
| `minGreenSteps` | Safety green phase step guard | `100` steps |

---

## 📁 Repository Directory Structure

```
catalyst_11/
├── 📁 agents/                 # Autonomous Reinforcement Learning Agent
│   └── agent.py               # Main Deep Q-Network Agent (Keras/TensorFlow + TraCI)
│
├── 📁 sumo_config/            # Microscopic SUMO Simulation Environment Configs
│   ├── RL.sumocfg             # Master SUMO configuration manifest
│   ├── catalystrl.net.xml     # Multi-lane road network & junction XML
│   ├── routes.rou.xml         # Microscopic vehicle demand and routes
│   ├── routes.add.xml         # E2 induction loop detector definitions
│   └── routes.netecfg         # Netedit project settings
│
├── 📄 .gitignore              # Git ignore configuration
├── 📄 RL.sumocfg              # Root SUMO config alias
├── 📄 catalystrl.net.xml      # Root network XML alias
├── 📄 routes.add.xml          # Root detector XML alias
├── 📄 routes.rou.xml          # Root routes XML alias
└── 📄 README.md               # Project Documentation
```

---

## ⚙️ Installation & Execution Guide

### 1. Prerequisites
- **Python 3.8+**
- **Eclipse SUMO** (Simulation of Urban MObility) v1.20+
- **TensorFlow / Keras**, **NumPy**, **Matplotlib**

Ensure `SUMO_HOME` environment variable is set:
```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
```

### 2. Environment Setup
```bash
# Clone the repository
git clone git@github.com:krishal77/catalyst-1.git
cd catalyst_11

# Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy matplotlib traci
```

### 3. Running the Deep Q-Network (DQN) Agent

```bash
python agents/agent.py
```

---

## 👥 Team & Credits

Developed with excellence by the **Catalyst11 Team**:
- 👨‍💻 **Krishnaram Thapaliya**
- 👩‍💻 **Anuska Acharya**
- 👩‍💻 **Prakriti Subedi**
- 👩‍💻 **Sulochana Subedi**

*Awarded 2nd Place in the Provincial Level Hackathon.*

---

## 📜 License

This project is open-source under the **MIT License**.
