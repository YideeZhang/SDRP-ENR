# FROZEN_MODEL_SPEC

This file is the frozen model specification for the current accepted formulation.

Current frozen version: `H-normal-single-charge`.

Its purpose is to preserve the exact modeling baseline currently used in the manuscript and solver-facing paper work.

Unless the user explicitly asks to revise the model, treat the content below as the authoritative reference for:

- notation
- objective
- derived quantities
- constraint blocks
- active structural assumptions

This frozen version follows the current manuscript baseline and is intended to remain aligned with the Transportation Research Part E / Part B writing target.

## 1. Scope

The active problem is a post-disaster electric truck-drone routing problem with microgrid support.

There are three node classes:

- `0`: depot
- `H`: microgrid service nodes, which also have material demand
- `C`: material demand nodes

The truck can directly visit:

- all `H` nodes
- the truck-accessible subset `C^T \subseteq C`

Drone missions serve material demand nodes in `C`.

The active drone modes are:

- star mode
- rendezvous mode

The active rendezvous formulation is the light-order version based on truck route order and successor gap `K`, not the older heavy-time mission-timing formulation.

## 2. Sets

\[
V : \text{set of trucks, indexed by } v
\]

\[
D_v : \text{set of drones carried by truck } v,\text{ indexed by } d
\]

\[
H : \text{set of microgrid service nodes, indexed by } h
\]

\[
C : \text{set of material demand nodes, indexed by } i,j,n
\]

\[
C^T \subseteq C : \text{truck-accessible demand nodes}
\]

\[
S = H \cup C
\]

\[
N^T = \{0\} \cup H \cup C^T
\]

\[
A^T \subseteq N^T \times N^T : \text{feasible truck arcs}
\]

\[
N^\star = H \cup C^T
\]

## 3. Parameters

\[
p_n : \text{population associated with node } n\in S
\]

\[
d_n : \text{material demand associated with node } n\in S
\]

\[
P_h : \text{population covered by microgrid node } h\in H
\]

\[
R_h : \text{restoration horizon of microgrid node } h\in H
\]

\[
P_h^o : \text{truck output power at microgrid node } h
\]

\[
\rho : \text{electricity-demand coefficient}
\]

\[
Q_v^T : \text{payload capacity of truck } v
\]

\[
w^D : \text{body weight of one drone}
\]

\[
q^D : \text{payload delivered by one drone sortie}
\]

\[
B_v : \text{battery capacity of truck } v
\]

\[
\underline{B}_v : \text{battery safety reserve of truck } v
\]

\[
B^D : \text{battery capacity of one drone}
\]

\[
t^{\max} : \text{maximum flight time of one drone sortie}
\]

\[
T^{\max} : \text{maximum duration of one truck mission}
\]

\[
v^T : \text{truck travel speed}
\]

\[
v^D : \text{drone flight speed}
\]

\[
e^T : \text{truck driving energy consumption per kilometer}
\]

\[
l_{ab} : \text{travel distance on arc } (a,b)\in A^T
\]

\[
\phi_{ab} : \text{road penalty factor on arc } (a,b)\in A^T
\]

\[
M^{big} : \text{sufficiently large constant}
\]

\[
K : \text{maximum successor gap in the light-order rendezvous formulation}
\]

## 4. Derived Quantities

\[
t_{ab}^v = \frac{l_{ab}}{v^T}\phi_{ab},
\qquad \forall (a,b)\in A^T,\ \forall v\in V
\]

\[
e_{ab}^v = e^T l_{ab}\phi_{ab},
\qquad \forall (a,b)\in A^T,\ \forall v\in V
\]

\[
t_{aia}^{fly,d} = \frac{2l_{ai}}{v^D},
\qquad \forall a\in N^\star,\ \forall i\in C,\ \forall d
\]

\[
e_{aia}^{d} = \frac{t_{aia}^{fly,d}}{t^{\max}} B^D,
\qquad \forall a\in N^\star,\ \forall i\in C,\ \forall d
\]

\[
t_{aib}^{fly,d} = \frac{l_{ai}+l_{ib}}{v^D},
\qquad \forall a,b\in N^T,\ \forall i\in C,\ \forall d
\]

\[
e_{aib}^{d} = \frac{t_{aib}^{fly,d}}{t^{\max}} B^D,
\qquad \forall a,b\in N^T,\ \forall i\in C,\ \forall d
\]

\[
E_h^{\mathrm{dem}} = P_h \rho R_h,
\qquad \forall h\in H
\]

\[
E_h^{\mathrm{sup}} = \sum_{v\in V} P_h^o \tau_h^v,
\qquad \forall h\in H
\]

## 5. Decision Variables

\[
x_{ab}^v \in \{0,1\},
\qquad \forall (a,b)\in A^T,\ \forall v\in V
\]

\[
z_h^v \in \{0,1\},
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\gamma_h^v \in \{0,1\},
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\eta_i \in \{0,1\},
\qquad \forall i\in C^T
\]

\[
\tau_h^v \ge 0,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\sigma_a^v \ge 0,
\qquad \forall a\in N^\star,\ \forall v\in V
\]

\[
A_a^v \ge 0,\quad L_a^v \ge 0,\quad w_a^v \ge 0,
\qquad \forall a\in N^T,\ \forall v\in V
\]

\[
T_v^{ret} \ge 0,
\qquad \forall v\in V
\]

\[
u_a^v \ge 0,
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
\ell_v \ge 0,
\qquad \forall v\in V
\]

\[
q_n \ge 0,\quad s_n \in [0,1],
\qquad \forall n\in S
\]

\[
q_n^{T,v} \ge 0,
\qquad \forall n\in H\cup C^T,\ \forall v\in V
\]

\[
y_{an}^{vd} \in \mathbb{Z}_+,
\qquad \forall a\in N^\star,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
r_{anb}^{vd} \in \{0,1\},
\qquad \forall a,b\in N^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
c_h \in [0,1],\quad g_h \in [0,1],
\qquad \forall h\in H
\]

\[
\chi_a^v : \text{visit-indicator expression used in route-order constraints}
\]

Under `H-normal-single-charge`, `z_h^v` indicates that truck `v`
physically visits or passes through `h`. The new variable `\gamma_h^v`
indicates that truck `v` is the charging-service truck at `h`.
The variable `\eta_i` indicates whether truck-accessible demand node `i`
is physically reached by at least one truck.

## 6. Objective

\[
\max Z
=
\alpha \sum_{h\in H} P_h g_h
+
\beta \sum_{n\in S} p_n s_n
\]

\[
\alpha = 1,\qquad \beta = 1,\qquad K = 3
\]

\[
g(\cdot)\ \text{has breakpoints } (0,0),\ (0.5,0.625),\ (1,1)
\]

## 7. Microgrid Energy Coverage

\[
E_h^{\mathrm{sup}} = E_h^{\mathrm{dem}} c_h,
\qquad \forall h\in H
\]

\[
0 \le c_h \le 1,
\qquad \forall h\in H
\]

\[
g_h = g(c_h),
\qquad \forall h\in H
\]

\[
t_h^{chg} : \text{arrival time of the truck selected to charge microgrid node } h
\]

\[
t_h^{chg} \ge A_h^v - M^{big}(1-\gamma_h^v),
\qquad \forall h\in H,\ \forall v\in V
\]

\[
t_h^{chg} \le A_h^v + M^{big}(1-\gamma_h^v),
\qquad \forall h\in H,\ \forall v\in V
\]

\[
R_h^{rem} = \max\{R_h - t_h^{chg},0\},
\qquad \forall h\in H
\]

\[
E_h^{arr} = P_h \rho R_h^{rem},
\qquad \forall h\in H
\]

\[
E_h^{\mathrm{sup}} \le E_h^{arr},
\qquad \forall h\in H
\]

## 8. Truck Routing and Time Constraints

\[
\sum_{b\in N^T\setminus\{0\}} x_{0b}^v = 1,
\qquad \forall v\in V
\]

\[
\sum_{a\in N^T\setminus\{0\}} x_{a0}^v = 1,
\qquad \forall v\in V
\]

\[
\sum_{b\in N^T,\ b\neq a} x_{ab}^v
=
\sum_{b\in N^T,\ b\neq a} x_{ba}^v,
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
\sum_{b\in N^T,\ b\neq h} x_{hb}^v = z_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\sum_{a\in N^T,\ a\neq h} x_{ah}^v = z_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\gamma_h^v \le z_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\sum_{v\in V} \gamma_h^v \le 1,
\qquad \forall h\in H
\]

\[
L_h^v = A_h^v + \tau_h^v + \sigma_h^v + w_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
L_i^v = A_i^v + \sigma_i^v + w_i^v,
\qquad \forall i\in C^T,\ \forall v\in V
\]

\[
A_b^v \ge L_a^v + t_{ab}^v - M^{big}(1-x_{ab}^v),
\qquad \forall (a,b)\in A^T,\ b\neq 0,\ \forall v\in V
\]

\[
T_v^{ret} \ge L_a^v + t_{a0}^v - M^{big}(1-x_{a0}^v),
\qquad \forall a\in N^T\setminus\{0\},\ \forall v\in V
\]

\[
T_v^{ret} \le T^{\max},
\qquad \forall v\in V
\]

\[
\tau_h^v \le T^{\max} \gamma_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
A_h^v \le T^{\max} z_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\sigma_h^v \le T^{\max} z_h^v,
\qquad \forall h\in H,\ \forall v\in V
\]

\[
\sigma_i^v \le T^{\max}\sum_{a\in N^T,\ a\neq i} x_{ai}^v,
\qquad \forall i\in C^T,\ \forall v\in V
\]

DFJ subtour elimination constraints are imposed for each truck route.

## 9. Truck Route Order Constraints

\[
\ell_v = \sum_{h\in H} z_h^v + \sum_{i\in C^T}\sum_{a\in N^T,\ a\neq i} x_{ai}^v,
\qquad \forall v\in V
\]

\[
u_a^v \ge \chi_a^v,
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
u_a^v \le |H\cup C^T|\,\chi_a^v,
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
u_a^v \le \ell_v,
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
\chi_a^v = z_a^v,
\qquad \forall a\in H,\ \forall v\in V
\]

\[
\chi_a^v = \sum_{b\in N^T,\ b\neq a} x_{ba}^v,
\qquad \forall a\in C^T,\ \forall v\in V
\]

\[
u_a^v \ge 1 - M^{big}(1-x_{0a}^v),
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
u_a^v \le 1 + M^{big}(1-x_{0a}^v),
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
\ell_v \ge u_a^v - M^{big}(1-x_{a0}^v),
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
\ell_v \le u_a^v + M^{big}(1-x_{a0}^v),
\qquad \forall a\in H\cup C^T,\ \forall v\in V
\]

\[
u_b^v \ge u_a^v + 1 - M^{big}(1-x_{ab}^v),
\qquad \forall a,b\in H\cup C^T,\ a\neq b,\ \forall v\in V
\]

\[
u_b^v \le u_a^v + 1 + M^{big}(1-x_{ab}^v),
\qquad \forall a,b\in H\cup C^T,\ a\neq b,\ \forall v\in V
\]

## 10. Material Balance and Coverage

\[
q_i
=
\sum_{v\in V} q_i^{T,v}
+
\sum_{v\in V}\sum_{d\in D_v}\sum_{a\in N^\star} q^D y_{ai}^{vd}
+
\sum_{v\in V}\sum_{d\in D_v}\sum_{a\in N^T}\sum_{b\in N^T} q^D r_{aib}^{vd},
\qquad \forall i\in C^T
\]

\[
q_j
=
\sum_{v\in V}\sum_{d\in D_v}\sum_{a\in N^\star} q^D y_{aj}^{vd}
+
\sum_{v\in V}\sum_{d\in D_v}\sum_{a\in N^T}\sum_{b\in N^T} q^D r_{ajb}^{vd},
\qquad \forall j\in C\setminus C^T
\]

\[
q_h = \sum_{v\in V} q_h^{T,v},
\qquad \forall h\in H
\]

\[
0 \le q_n \le d_n,
\qquad \forall n\in S
\]

\[
q_n = d_n s_n,
\qquad \forall n\in S
\]

## 11. Truck Capacity Constraints

\[
\sum_{n\in H\cup C^T} q_n^{T,v}
+
\sum_{d\in D_v}\sum_{a\in N^\star}\sum_{n\in C} q^D y_{an}^{vd}
+
\sum_{d\in D_v}\sum_{a\in N^T}\sum_{n\in C}\sum_{b\in N^T} q^D r_{anb}^{vd}
\le Q_v^T - |D_v|w^D,
\qquad \forall v\in V
\]

\[
q_n^{T,v}
\le
d_n \chi_n^v,
\qquad \forall n\in H\cup C^T,\ \forall v\in V
\]

\[
\eta_i \ge \chi_i^v,
\qquad \forall i\in C^T,\ \forall v\in V
\]

\[
\eta_i \le \sum_{v\in V}\chi_i^v,
\qquad \forall i\in C^T
\]

Drone sorties are used only for truck-accessible ordinary demand nodes that are not physically reached by any truck.

\[
y_{ai}^{vd} \le M^{big}(1-\eta_i),
\qquad \forall i\in C^T,\ \forall a\in N^\star,\ \forall v\in V,\ \forall d\in D_v
\]

\[
r_{aib}^{vd} \le 1-\eta_i,
\qquad \forall i\in C^T,\ \forall a,b\in N^T,\ \forall v\in V,\ \forall d\in D_v
\]

## 12. Star-Mode Constraints

\[
y_{hn}^{vd} \le M^{big} z_h^v,
\qquad \forall h\in H,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
y_{an}^{vd}
\le
M^{big}\sum_{b\in N^T,\ b\neq a} x_{ba}^v,
\qquad \forall a\in C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
\sum_{n\in C} t_{hnh}^{fly,d} y_{hn}^{vd}
\le
\tau_h^v + \sigma_h^v,
\qquad \forall h\in H,\ \forall v\in V,\ \forall d\in D_v
\]

\[
\sum_{n\in C} t_{ana}^{fly,d} y_{an}^{vd}
\le
\sigma_a^v,
\qquad \forall a\in C^T,\ \forall v\in V,\ \forall d\in D_v
\]

\[
e_{ana}^{d} y_{an}^{vd} \le B^D y_{an}^{vd},
\qquad \forall a\in N^\star,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

Frozen interpretation:

- drones in star mode serve only nodes in `C`
- if the launch node is `h\in H`, star completion must fit inside `\tau_h^v + \sigma_h^v`
- if the launch node is `a\in C^T`, star completion must fit inside `\sigma_a^v`

## 13. Rendezvous-Mode Constraints

The active rendezvous formulation is the simplified light-order formulation.

It does not use the older dedicated mission timing variables such as:

- `rstart`
- `rend`
- explicit rendezvous-only waiting variables

Instead:

- launch timing is anchored to truck departure time `L_a^v`
- reunion feasibility is enforced through truck waiting / departure variables
- structural feasibility is controlled by route order variables `u` and successor gap `K`

\[
r_{anb}^{vd}
\le
\sum_{j\in N^T,\ j\neq a} x_{aj}^v,
\qquad \forall a,b\in N^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
r_{anb}^{vd}
\le
\sum_{j\in N^T,\ j\neq b} x_{jb}^v,
\qquad \forall a,b\in N^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
u_b^v \ge 1 - M^{big}(1-r_{0nb}^{vd}),
\qquad \forall b\in H\cup C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
u_b^v \le K + M^{big}(1-r_{0nb}^{vd}),
\qquad \forall b\in H\cup C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
\ell_v \le u_a^v + K - 1 + M^{big}(1-r_{an0}^{vd}),
\qquad \forall a\in H\cup C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
u_b^v \ge u_a^v + 1 - M^{big}(1-r_{anb}^{vd}),
\qquad \forall a,b\in H\cup C^T,\ a\neq b,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
u_b^v \le u_a^v + K + M^{big}(1-r_{anb}^{vd}),
\qquad \forall a,b\in H\cup C^T,\ a\neq b,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
T_v^{ret} \ge L_a^v + t_{an0}^{fly,d} - M^{big}(1-r_{an0}^{vd}),
\qquad \forall a\in H\cup C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
w_b^v \ge L_a^v + t_{anb}^{fly,d} - A_b^v - \tau_b^v - \sigma_b^v - M^{big}(1-r_{anb}^{vd}),
\qquad \forall a\in N^T,\ \forall b\in H,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
L_b^v \ge L_a^v + t_{anb}^{fly,d} - M^{big}(1-r_{anb}^{vd}),
\qquad \forall a\in N^T,\ \forall b\in H,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
w_b^v \ge L_a^v + t_{anb}^{fly,d} - A_b^v - \sigma_b^v - M^{big}(1-r_{anb}^{vd}),
\qquad \forall a\in N^T,\ \forall b\in C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
L_b^v \ge L_a^v + t_{anb}^{fly,d} - M^{big}(1-r_{anb}^{vd}),
\qquad \forall a\in N^T,\ \forall b\in C^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
t_{anb}^{fly,d}
\le
t^{\max} + M^{big}(1-r_{anb}^{vd}),
\qquad \forall a,b\in N^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
e_{anb}^{d}
\le
B^D + M^{big}(1-r_{anb}^{vd}),
\qquad \forall a,b\in N^T,\ \forall n\in C,\ \forall v\in V,\ \forall d\in D_v
\]

\[
\sum_{n\in C}\sum_{b\in N^T} r_{anb}^{vd} \le 1,
\qquad \forall a\in N^T,\ \forall v\in V,\ \forall d\in D_v
\]

\[
\sum_{a\in N^T}\sum_{n\in C} r_{anb}^{vd} \le 1,
\qquad \forall b\in N^T,\ \forall v\in V,\ \forall d\in D_v
\]

Frozen interpretation:

- rendezvous missions serve only nodes in `C`
- `K=3` is the active structural window
- if launch is from depot, the recovery node must lie within the first `K` visited non-depot truck nodes
- if recovery is depot, the launch node must lie within the last `K` visited non-depot truck nodes
- if both launch and recovery are non-depot truck nodes, the recovery order must lie within `[u_a^v + 1,\ u_a^v + K]`
- truck waiting variable `w_b^v` absorbs reunion waiting when recovery is at `H` or `C^T`

## 14. Shared Battery Constraints

\[
E_v^{\mathrm{drive}} = \sum_{(a,b)\in A^T} e_{ab}^v x_{ab}^v,
\qquad \forall v\in V
\]

\[
E_v^{\mathrm{drone}}
=
\sum_{d\in D_v}\sum_{a\in N^\star}\sum_{n\in C} e_{ana}^{d} y_{an}^{vd}
+
\sum_{d\in D_v}\sum_{a\in N^T}\sum_{n\in C}\sum_{b\in N^T} e_{anb}^{d} r_{anb}^{vd},
\qquad \forall v\in V
\]

\[
E_v^{\mathrm{grid}} = \sum_{h\in H} P_h^o \tau_h^v,
\qquad \forall v\in V
\]

\[
E_v^{\mathrm{drive}} + E_v^{\mathrm{drone}} + E_v^{\mathrm{grid}}
\le
B_v - \underline{B}_v,
\qquad \forall v\in V
\]

## 15. Frozen Assumptions

The following assumptions are currently frozen and should not be silently changed:

1. `H` nodes contribute to both energy-service coverage and material-service coverage.
2. Multiple trucks may physically visit or pass through the same `H` node.
3. At most one truck can be selected as the charging-service truck for each `H` node through `\gamma_h^v`.
4. `\tau_h^v` is positive only when `\gamma_h^v=1`; physical H visits without charging have `\tau_h^v=0`.
5. The arrival-aware microgrid demand cap uses the charging-service arrival time `t_h^{chg}`, not the sum of all physical H visit arrival times.
6. The same truck should not visit the same `H` node more than once in one route.
7. Drone deliveries, whether in star mode or rendezvous mode, are restricted to `C`.
8. `C^T` is the truck-accessible subset of `C`.
9. Star launches are allowed only from `H` and `C^T`.
10. At `H`, star-mode completion uses the combined stay budget `\tau_h^v + \sigma_h^v`.
11. At `C^T`, star-mode completion uses only `\sigma_a^v`.
12. Rendezvous uses the light-order formulation with `K=3`.
13. Truck route order is represented by `u_a^v` and route length `\ell_v`.
14. The older explicit rendezvous mission-timing formulation is inactive.
15. Shared battery accounting includes truck driving, drone operation, and microgrid energy supply.

## 16. Primary Synchronization Reference

For paper writing and future model edits, use the following interpretation:

- `A_a^v` is truck arrival time
- `L_a^v` is truck departure time
- `w_a^v` is truck waiting time
- `\tau_h^v` is charging / grid-service stay at `H`
- `\sigma_a^v` is extra stay reserved for star-related completion

Under the current baseline, rendezvous recovery timing is enforced indirectly through:

- `T_v^{ret}` when recovery is at depot
- `w_b^v` and `L_b^v` when recovery is at `H` or `C^T`

This is the formulation that should be treated as current unless a later user instruction explicitly replaces it.
