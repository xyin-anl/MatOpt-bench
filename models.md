# Mathematical Formulations of MatOpt Benchmark Models

This document provides detailed mathematical formulations for all 11 optimization models in the MatOpt benchmark suite. All formulations use LaTeX rendering compatible with GitHub markdown.

## Table of Contents

1. [Monometallic Nanocluster Design](#1-monometallic-nanocluster-design)
2. [Bimetallic Nanocluster Design](#2-bimetallic-nanocluster-design)
3. [Nanopatterned Surface Design](#3-nanopatterned-surface-design)
4. [Bifunctional Surface Design](#4-bifunctional-surface-design)
5. [Metal Oxide Bulk Design (Oxygen Vacancy)](#5-metal-oxide-bulk-design-oxygen-vacancy)
6. [Ionic Crystal Structure Prediction](#6-ionic-crystal-structure-prediction)
7. [Nanowire Design](#7-nanowire-design)
8. [Cluster Expansion Optimization](#8-cluster-expansion-optimization)
9. [Solid Solution QUBO Design](#9-solid-solution-qubo-design)
10. [Crystal Structure Prediction with FM](#10-crystal-structure-prediction-with-fm)
11. [High-Entropy Alloy Design with FM](#11-high-entropy-alloy-design-with-fm)

---

## 1. Monometallic Nanocluster Design

**Objective**: Maximize cohesive energy of a monometallic nanocluster

**Mathematical Formulation**:

$$\max_{y,CN,\lambda} \quad E_{coh} = \frac{E_{coh}^{bulk}}{N} \sum_{i \in Sites} \sqrt{\frac{CN_i}{CN_{max}}}$$

Subject to:
$$\sum_{i \in Sites} y_i = N \quad \text{(cluster size)}$$

$$CN_i = \sum_{j \in N(i)} y_j \quad \forall i \in Sites \quad \text{(coordination number)}$$

$$\sqrt{\frac{CN_i}{CN_{max}}} = \sum_{k=0}^{K} \lambda_{ik} \cdot r_k \quad \forall i \quad \text{(piecewise linear approx.)}$$

$$\sum_{k=0}^{K} \lambda_{ik} = y_i \quad \forall i \quad \text{(convexity constraint)}$$

$$\lambda_{i,k} \leq \lambda_{i,k-1} \quad \forall i, k > 0 \quad \text{(ordering constraint)}$$

$$y_i \in \{0,1\}, \quad 0 \leq CN_i \leq CN_{max}, \quad \lambda_{ik} \geq 0$$

**Key Features**:
- Binary variables: $N$ (site occupancy)
- Continuous variables: $N$ (coordination) + $N \times K$ (piecewise weights)
- Quadratic terms: None (linearized via piecewise approximation)

---

## 2. Bimetallic Nanocluster Design

**Two-Step Approach**:
1. First optimize monometallic shape (using formulation from Example 1)
2. Then optimize bimetallic composition on fixed shape

**Step 2 Formulation**:

$$\max_{x} \quad E_{coh} = -\sum_{i \in Sites} \sum_{j \in N(i)} \sum_{k \in Elements} \sum_{l \in Elements} x_{ik} \cdot x_{jl} \cdot G_{kl}(CN_i, CN_j)$$

Subject to:
$$\sum_{k \in Elements} x_{ik} = 1 \quad \forall i \in FixedSites \quad \text{(one atom per site)}$$

$$\sum_{i \in Sites} x_{ik} = N_k \quad \forall k \in Elements \quad \text{(composition constraint)}$$

$$x_{ik} \in \{0,1\}$$

**Linearization** (for MILP):
$$z_{ijkl} \geq x_{ik} + x_{jl} - 1 \quad \forall i,j,k,l$$

$$z_{ijkl} \leq x_{ik}, \quad z_{ijkl} \leq x_{jl} \quad \forall i,j,k,l$$

**Key Features**:
- Binary variables: $|Sites| \times |Elements|$
- Quadratic terms: $O(|Sites|^2 \times |Elements|^2)$ (linearized with z variables)

---

## 3. Nanopatterned Surface Design

**Objective**: Balance catalytic activity and stability

$$\max_{y,CN,GCN,I} \quad f = w \cdot \frac{\sum_{i \in Surface} I_i}{|Surface|} + (1-w) \cdot \left(-\sum_{i \in Surface} \sqrt{\frac{CN_i}{CN_{bulk}}}\right)$$

Subject to:
$$y_i = 1 \quad \forall i \in FixedLayers \quad \text{(fixed bottom layers)}$$

$$y_i \leq \sum_{j \in Supports(i)} y_j \quad \forall i \notin FixedLayers \quad \text{(support constraint)}$$

$$CN_i = \sum_{j \in N(i)} y_j \quad \forall i$$

$$GCN_i = \sum_{j \in N(i)} y_j \cdot \frac{CN_j}{CN_{bulk}} \quad \forall i$$

$$I_i \geq 1 - M(1 - y_i) \quad \forall i : CN_i = CN_{target} \land GCN_{min} \leq GCN_i \leq GCN_{max}$$

$$y_i, I_i \in \{0,1\}, \quad 0 \leq CN_i \leq CN_{bulk}, \quad 0 \leq GCN_i \leq CN_{bulk}$$

**Key Features**:
- Binary variables: $2 \times |Sites|$ (occupancy + ideal sites)
- Continuous variables: $2 \times |Sites|$ (CN + GCN)
- Quadratic terms: $O(|Sites|^2)$ in GCN calculation (linearized)

---

## 4. Bifunctional Surface Design

**Objective**: Maximize combined activity from configuration pairs

$$\max_{x,z} \quad \sum_{c \in Configs} \sum_{d \in Configs} z_{cd} \cdot F(c,d)$$

Subject to:
$$\sum_{c \in Configs} x_c = K \quad \text{(select K configurations)}$$

$$z_{cd} \leq x_c, \quad z_{cd} \leq x_d \quad \forall c,d$$

$$z_{cd} \geq x_c + x_d - 1 \quad \forall c,d$$

$$x_c, z_{cd} \in \{0,1\}$$

**Key Features**:
- Binary variables: $|Configs| + |Configs|^2$
- Quadratic terms: $|Configs|^2$ (linearized with z variables)

---

## 5. Metal Oxide Bulk Design (Oxygen Vacancy)

**Objective**: Maximize oxygen vacancy formation activity

$$\max_{x,d} \quad \sum_{c \in VacancyConfigs} x_c \cdot E_c$$

Subject to:
$$\sum_{c \in VacancyConfigs} x_c = K \quad \text{(select K configurations)}$$

$$p_{min} \leq \frac{\sum_{i \in Region} d_i}{|Region|} \leq p_{max} \quad \forall Region \quad \text{(local doping)}$$

$$g_{min} \leq \frac{\sum_{i \in BSites} d_i}{|BSites|} \leq g_{max} \quad \text{(global doping)}$$

$$d_i = 1 \quad \forall i \in InSites(c), \forall c : x_c = 1 \quad \text{(dopant consistency)}$$

$$x_c, d_i \in \{0,1\}$$

**Key Features**:
- Binary variables: $|Configs| + |BSites|$
- Quadratic terms: None

---

## 6. Ionic Crystal Structure Prediction

**Objective**: Minimize total electrostatic energy

$$\min_{y,z} \quad E = E_{Coulomb} + E_{Buckingham}$$

where:
$$E_{Coulomb} = \sum_{i \in Sites} \sum_{j \in Sites} \sum_{k \in Ions} \sum_{l \in Ions} z_{ijkl} \cdot C2_{ijkl}$$

$$E_{Buckingham} = \sum_{(i,j) \in ShortRange} \sum_{k \in Ions} \sum_{l \in Ions} z_{ijkl} \cdot B_{kl}(r_{ij})$$

Subject to:
$$\sum_{k \in Ions} y_{ik} = 1 \quad \forall i \in Sites$$

$$\sum_{i \in Orbit} y_{ik} = n_k^{orbit} \quad \forall k \in Ions, \forall Orbit$$

$$\sum_{i \in Sites} \sum_{k \in Ions} q_k \cdot y_{ik} = 0 \quad \text{(charge neutrality)}$$

$$z_{ijkl} \geq y_{ik} + y_{jl} - 1, \quad z_{ijkl} \leq y_{ik}, \quad z_{ijkl} \leq y_{jl}$$

$$y_{ik}, z_{ijkl} \in \{0,1\}$$

**Key Features**:
- Binary variables: $|Sites| \times |Ions| + |Sites|^2 \times |Ions|^2$
- Quadratic terms: $O(|Sites|^2 \times |Ions|^2)$ (linearized)

---

## 7. Nanowire Design

**Objective**: Maximize cohesive energy of cylindrical nanowire

$$\max_{y,CN} \quad E_{coh} = \sum_{i \in Sites} y_i \cdot \sqrt{\frac{CN_i}{CN_{max}}}$$

Subject to:
$$y_i = 1 \quad \forall i \in Core \quad \text{(fixed core atoms)}$$

$$r_i \leq 0.2 \cdot R \quad \forall i \in Core \quad $$

$$y_i \leq \sum_{j \in PathToCore(i)} y_j \quad \forall i \notin Core \quad \text{(connectivity)}$$

$$CN_i = \sum_{j \in N_{PBC}(i)} y_j \quad \forall i \quad \text{(with periodic boundaries)}$$

$$y_i \in \{0,1\}, \quad 0 \leq CN_i \leq CN_{max}$$

**Key Features**:
- Binary variables: $|Sites|$
- Continuous variables: $|Sites|$ (coordination)
- Quadratic terms: None (square root linearized)

---

## 8. Cluster Expansion Optimization

**Objective**: Multi-property optimization using ML cluster expansion

$$\min_{x} \quad \sum_{p \in Properties} \alpha_p \cdot ECE_p(x)$$

where:
$$ECE_p(x) = J_{0,p} + \sum_{\gamma \in Clusters} J_{\gamma,p} \cdot f_\gamma(x)$$

$$f_\gamma(x) = \prod_{i \in \gamma} \sigma_i(x), \quad \sigma_i(x) = 2x_{i,element} - 1$$

Subject to:
$$\sum_{k \in Elements} x_{ik} = 1 \quad \forall i \in Sites$$

$$c_k^{min} \leq \frac{\sum_{i \in Sites} x_{ik}}{|Sites|} \leq c_k^{max} \quad \forall k$$

$$x_{ik} \in \{0,1\}$$

**Linearization** of cluster functions requires auxiliary variables for each cluster product.

**Key Features**:
- Binary variables: $|Sites| \times |Elements|$
- Quadratic/higher-order terms: Depends on cluster sizes (typically up to 4-body)

---

## 9. Solid Solution QUBO Design

**Objective**: Minimize QUBO energy

$$\min_{x} \quad E = x^T Q x + \mu \sum_{i \in Sites} x_i$$

Subject to (constrained mode):
$$\sum_{i \in Sites} x_i = K \quad \text{(fixed number of substitutions)}$$

$$x_i = 0 \quad \forall i \in FrozenSites$$

$$x_i \in \{0,1\}$$

**Key Features**:
- Binary variables: $|Sites|$
- Quadratic terms: $|Sites|^2$ (from QUBO matrix Q)

---

## 10. Crystal Structure Prediction with FM

**Objective**: Minimize factorization machine energy

$$\min_{x} \quad E = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

where $\langle v_i, v_j \rangle = \sum_{k=1}^{r} v_{ik} \cdot v_{jk}$ (rank-r factorization)

Subject to:
$$\sum_{i \in Group_m} x_i = 1 \quad \forall m \in PropertyGroups \quad \text{(one-hot encoding)}$$

$$x_i \in \{0,1\}$$

**Key Features**:
- Binary variables: Total bits in encoding (typically 100-1000)
- Quadratic terms: $O(n^2)$ where n is number of bits

---

## 11. High-Entropy Alloy Design with FM

**Objective**: Minimize free energy at finite temperature

$$\min_{x,c} \quad F = E_{FM}(x) - T \cdot S_{config}(c)$$

where:
$$E_{FM}(x) = w_0 + \sum_{i,k} w_{ik} x_{ik} + \sum_{i,j,k,l} \langle v_{ik}, v_{jl} \rangle x_{ik} x_{jl}$$

$$S_{config}(c) = -R \sum_{k \in Elements} c_k \ln(c_k)$$

$$c_k = \frac{\sum_{i \in Sites} x_{ik}}{|Sites|}$$

Subject to:
$$\sum_{k \in Elements} x_{ik} = 1 \quad \forall i \in Sites$$

$$c_k \leq 0.6 \quad \forall k \in Elements \quad$$

$$x_{ik} \in \{0,1\}, \quad 0 \leq c_k \leq 1$$

**Key Features**:
- Binary variables: $|Sites| \times |Elements|$
- Continuous variables: $|Elements|$ (compositions)
- Quadratic terms: $O(|Sites|^2 \times |Elements|^2)$

---

## Model Complexity Summary

| Example | Model Name | Binary Vars | Continuous Vars | Quadratic Terms |
|---------|------------|-------------|-----------------|-----------------|
| 1 | Monometallic Cluster | $N$ | $N(K+1)$ | 0 |
| 2 | Bimetallic Cluster | $NK$ | 0 | $N^2K^2$ |
| 3 | Surface Design | $2N$ | $2N$ | $N^2$ |
| 4 | Bifunctional Surface | $C + C^2$ | 0 | $C^2$ |
| 5 | Oxygen Vacancy | $C + B$ | 0 | 0 |
| 6 | Ionic Crystal (IPCSP) | $NK + N^2K^2$ | 0 | $N^2K^2$ |
| 7 | Nanowire Design | $N$ | $N$ | 0 |
| 8 | Cluster Expansion | $NK$ | Aux. vars | Cluster-dependent |
| 9 | Solid Solution QUBO | $N$ | 0 | $N^2$ |
| 10 | Crystal FM | $B$ | 0 | $B^2$ |
| 11 | HEA with FM | $NK$ | $K$ | $N^2K^2$ |

**Legend**:
- $N$: Number of sites/atoms
- $K$: Number of element types or piecewise segments
- $C$: Number of configurations
- $B$: Number of bits in encoding
- Aux. vars: Auxiliary variables for linearization

**Notes**:
1. Examples 9, 10, 11 can be solved as QUBO/quadratic problems directly or linearized for MILP
2. Linearization typically introduces additional binary variables equal to the number of quadratic terms
3. The actual problem size depends on specific instance parameters (lattice size, number of elements, etc.)