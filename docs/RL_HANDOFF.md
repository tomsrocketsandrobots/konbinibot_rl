# RL / Sim2Real Workspace: Handoff

**Written:** 2026-10-13. Written on the robot's LattePanda (`konbinipanda`) at the end of a planning session.

**Purpose:** start a **new, separate workspace on the RTX 3080 PC** for training Konbini's controller with reinforcement learning in simulation, then transferring it to the real robot. This doc carries the decisions, robot facts, and starting plan from that session. Read it top to bottom once, then use §8 (milestones) and §9 (M0 checklist) to begin.

The model-based design that came before this lives in the `konbinibot` repo at git tag **`pre-rl-pivot`**.

---

## 1. Decision and rationale

**Decision:** replace the model-based control stack with a **learned policy**, trained in a GPU-parallel simulator with domain randomization and transferred sim2real. The replaced stack is the MATLAB-designed LPV-LQR, the planned MPC, and STA-SMC.

**Why:**
- The robot is complex: closed-chain 8-bar legs, neck, 3-DOF head, and a cable-driven continuum tail.
- The hand-derived model accumulated unresolved problems:
  - a pitch sign convention that is unverified (it looks inverted between `ode_fun_p2.m`, the docs, and the sim config)
  - gain-schedule grid and rate mismatches
  - contradictory Phase 2.5 results
  - see `konbinibot/plans/ISSUES.md` and README §4
- Wheeled-legged balancing is a well-proven sim2real RL problem.

**What is *not* changing:**
- **The ELEGNT-style behavior goal.** The expressive behavior layer (functional utility F + γ·expressive utility E, motion primitives, gaze, telegraphing) stays as the top layer.
- **The hardware.** All hardware bring-up in `konbinibot` stays valid and is still needed.

---

## 2. Target architecture

```
┌───────────────────────────────────────────────────────────────┐
│ BEHAVIOR LAYER  (ELEGNT "Beta" layer: kept)                     │
│ primitive selection, phase sequencing, γ, gaze targets          │
│ → an animation / reference engine (BD-X style) that composes    │
│   and blends motion references + command signals                │
└──────────────────────────┬────────────────────────────────────┘
                           │ commands (v, yaw rate, leg height, lean,
                           │ neck, tail, style…) + reference poses
┌──────────────────────────▼────────────────────────────────────┐
│ LEARNED POLICY  (replaces LQR / MPC / SMC: "Alpha")            │
│ command- and reference-conditioned; trained in sim; MLP → ONNX │
└──────────────────────────┬────────────────────────────────────┘
                           │ joint targets / torques @ policy rate
┌──────────────────────────▼────────────────────────────────────┐
│ konbinibot ros2_control (400 Hz) on LattePanda 3 Delta          │
│ CyberGear wheels · MyActuator hips · Dynamixel neck + head      │
└───────────────────────────────────────────────────────────────┘
```

- **Pattern to follow: Disney's BD-X**, "Design and Control of a Bipedal Robotic Character" (`konbinibot/refs/bd_x_paper.md`):
  - Animators and procedural generators author motion.
  - An animation engine produces command and reference signals.
  - RL policies are trained to track those references robustly while balancing.
  - Different motion types may get separate policies.
- **ELEGNT** (`konbinibot/refs/ELEGNT_PAPER.MD`) supplies the behavior framing: R = F + γE, with the expression dimensions intention, attention, attitude, and affect, built from kinesics and proxemics primitives. In ELEGNT the "MDP" is a design framing; the movements were designed, not learned. Konbini's version of this spec is `konbinibot/plans/control_architecture.md` §9, which is reusable almost as-is.
- **§9 fields that need redefining**, because they were defined in MPC terms:
  - `w` (per-DOF weights): becomes "which DOFs the reference moves".
  - `j_budget`: becomes reference timing and smoothing, or a style command.
  - `p_mode`: becomes "wheels roll" vs. "contact fixed", expressed as a command or as reference content.
- **Head yaw / pitch / roll** can stay outside the policy as Dynamixel position servos driven by the behavior layer, as in §9's head controller. The policy should still *observe* them.
- **The neck** is in the balance-relevant set, so it is policy-owned or policy-tracked.
- **The tail is both functional and expressive.**
  - The policy owns it for balance.
  - The behavior layer requests expressive tail motion as a reference that the policy tracks when balance allows.
  - This is the F vs. γE trade-off happening at runtime.
- **Deployment:**
  - Export the policy to ONNX.
  - Run it in a new ros2_control controller package in `konbinibot` (onnxruntime).
  - The policy runs at roughly 50–100 Hz on top of the 400 Hz hardware loop.
  - A small MLP is cheap on the LattePanda CPU.
  - Keep the existing `balance_controller`, or a simple one, as a safe-mode fallback.

**The most important early design decision:** the command and reference interface between the behavior layer and the policy. It defines what the policy trains on, what the reference engine produces, and what ROS plumbs through. Draft it during M2; freeze it before M6.

---

## 3. Robot facts

Canonical sources in `konbinibot`:
- `plans/README.md` §3, with a full table and source lines
- `sim/balance/P2/ricket_balance_phase2.m` (params)
- `sim/tail/tail_params.m`
- `src/bot_description/urdf/*.xacro`

### Actuators (all torques per motor unless noted)

| Joint(s) | Actuator | Interface | Key specs (from model params; verify on bench) |
|---|---|---|---|
| left/right wheel | CyberGear | CAN `can0`, 29-bit ext IDs **1 / 2**, torque mode (MIT-style, kp = kd = 0) | stall 6.0 N·m; corner 23 rad/s; no-load 62 rad/s. Wheel radius **0.072 m** |
| left/right hip | **MyActuator X10 S2 V3** | CAN `can0`, 11-bit, motor IDs **11 / 12** (frames 0x14B/C cmd, 0x24B/C reply) | Kt 7.46 N·m/A. Model used stall 50.39 N·m, corner 5.24 rad/s, no-load 5.61 rad/s (these predate the X10 identification, so verify) |
| neck (`neck_ext_joint`) | Dynamixel XH540-W270-R through a four-bar (N ≈ 2, *estimated*) | USB serial `/dev/dynamixel_u2d2` | motor stall 9.9 N·m (≈ 19.8 N·m at the joint); 4.08 rad/s no-load at the motor. Runs its own internal position loop |
| head yaw / pitch / roll | Dynamixel | same bus | Position servos |
| tail (2 segments × bending axes) | TBD motors + pulley, cable-driven | — | model: τ_max 3.0 N·m, pulley r 0.015 m, cable offset 0.010 m |

### Sensors (simulate all with realistic noise, rate, latency, and dropout; the policy uses proprioception only)

| Sensor | Details |
|---|---|
| Body IMU | Xsens **MTi-630** over CAN, **400 Hz**, VRU mode (no mag), `imu_link` (parent `robot_frame_link`, xyz 0.1 −0.01 0.05). Quaternion, rate of turn, FreeAcceleration (gravity removed on-chip, earth frame), hardware CAN timestamps. ⚠ The smoke-test log shows accel z = 9.78 m/s², which suggests it predates the switch to FreeAcceleration. Re-check before trusting the gravity-removed signal |
| Joint encoders | Wheel (CyberGear), hip (MyActuator), neck + head (Dynamixel: position, velocity, current) |
| Motor current / torque | All motor families report it; useful for observations and for actuator system identification |
| Head lidar | RPLIDAR C1 (planned). **Not a policy input:** feeds perception / SLAM / heightmap |
| Head camera | Planned. **Not a policy input** (VRAM / sim-speed cost) |
| Head IMU | BNO055 if found, else MPU-6050 + LSM303DLHC on a Teensy 4.1 (micro-ROS). Planned |
| ToF / contact | Mentioned in the old architecture's estimation layer; status unknown |

### Mass and geometry (simplification is acceptable)

- **Masses:**
  - body 6.8 kg
  - legs 0.864 kg × 2
  - wheels 0.345 kg × 2
  - head 3.02 kg (lumped)
  - neck link 0.05 kg (0.30 m long, pivot 0.18 m forward of the hip pivot)
  - total ≈ 12.3 kg without the tail
- The body CoM was approximated at the hip pivot. The tail study found the body pitch inertia `M_theta` ≈ 0.587 kg·m², with the head dominating, and found the bare URDF inertia **16.6× too small**. **Check URDF inertias before trusting them in sim.**
- **URDF / xacro:** `src/bot_description/urdf/konbini.urdf.xacro`. Meshes: `chassis.stl`, `head.stl`, `wheel2.stl` in `src/bot_description/meshes/`.
- **Leg:** a closed 8-bar linkage driven by one hip motor.
  - Validated forward kinematics, Jacobians, and effective inertia are in `sim/leg/leg_linkage.m`.
  - Masses and inertias are in `sim/params/linkage_params.m`.
  - The geometry source is `sim/data/leg_geo7B.linkage2`.
  - The hip angle φ operating range has two usable arcs: speed arc ≈ 0.42–0.75 rad and force arc ≈ 2.38–6.10 rad, with an unusable gap between them. Nominal φ is 4.7554 rad.
- **Tail:** a two-segment cable-driven continuum.
  - Arc lengths 0.18 m / 0.15 m, masses 0.12 kg / 0.10 kg.
  - Curvature limit 24 rad/m.
  - Mount at [−0.15, −0.10] m (x, z) from the hip pivot.
  - EI_eff 0.002 N·m² and b_eff 0.001 are **placeholders that must be measured on a prototype segment**.
  - Model and derivation: `konbinibot/plans/tail_model_plan.md`, `sim/tail/`.
- **Conventions:** REP 103 (x forward, z up). ⚠ The pitch sign convention was never settled in the old model; define it once, explicitly, in the new one.

### Compute

- **Training:** RTX 3080 (10 GB) and 32 GB RAM, run headless. Enough for proprioceptive training with thousands of parallel environments. Not enough for rendered-camera RL at scale.
- **Robot:** LattePanda 3 Delta, 4 cores, Ubuntu with PREEMPT_RT, ROS 2 Kilted.

---

## 4. Modeling approach in sim

- **Simulator: start with MuJoCo + MuJoCo Playground** (JAX / Brax PPO). Reasons:
  - native closed-loop (equality) constraints for the leg
  - tendons and joint-coupling for the tail
  - lighter install
  - comfortable on 10 GB of VRAM

  Isaac Lab, with its larger locomotion ecosystem, is the alternative; re-evaluate after M1 if needed. Verify current versions and install steps at M0; this space moves fast.
- **Leg:** either keep the real 8-bar with equality constraints, or collapse it to an equivalent serial / virtual leg driven by `leg_linkage` lookup tables. Prefer the simpler option first.
- **Tail:** approximate each segment as 4–8 small hinges coupled to share one angle. That is the constant-curvature assumption, enforced with joint equality constraints. Add hinge stiffness and damping for EI_eff and b_eff, and drive each bending axis through a tendon or gear. Validate against the derived EOM in `sim/tail/tail_eom_*`.
- **Actuators, where sim2real is decided:** model the torque–speed limits, latency (CAN round trip), friction, and the Dynamixel internal position loop. Fit from bench data; `konbinibot/scripts/myactuator_torque_sweep.py` is a data source. Randomize what is uncertain.
- **Domain randomization:** masses, CoM offsets (especially head and tail), friction, motor strength, latency, sensor noise and bias, terrain slope.
- **Teacher / student:** a teacher may use privileged information (true CoM velocity, friction, slope); a student uses only real sensors. This removes the need to settle the old "is the IMU fused into the EKF" estimation question up front.

---

## 5. What to reuse from `konbinibot`

| Need | Where |
|---|---|
| Behavior / expression spec | `plans/control_architecture.md` §9 (and §2, §16) |
| Robot description | `src/bot_description/urdf/`, `meshes/` |
| Leg kinematics | `sim/leg/leg_linkage.m`, `sim/params/linkage_params.m`, `sim/data/leg_geo7B.linkage2` |
| Tail model + parameters | `plans/tail_model_plan.md`, `sim/tail/tail_params.m`, `sim/tail/tail_eom_*` |
| Physical parameters | `sim/balance/P2/ricket_balance_phase2.m` (params block), `plans/README.md` §3 |
| Hardware facts, open hardware issues | `plans/README.md`, `plans/ISSUES.md` |
| Reference papers | `refs/ELEGNT_PAPER.MD`, `refs/bd_x_paper.md`, `refs/paper1.md` (model-based wheeled-biped contrast) |
| Bench scripts (actuator data) | `scripts/myactuator_*.py` |

---

## 6. How the two workspaces relate

- **`konbinibot`** (LattePanda, ROS 2): hardware bring-up and, later, deployment.
  - The hardware track continues there: hip bench tests, `hardware.launch.py`, the hip xacro fix, wheel odometry, and the safe-mode fallback.
  - It eventually gets a `policy_controller` package.
- **New RL workspace** (3080 PC): MJCF model, environments, rewards, training, evaluation, and ONNX export.
- **Shared contract:** the robot description and the policy's observation / action / command specification (names, units, order, rates, normalization). Version this spec.

---

## 7. Planning approach (to avoid another 20-doc pile)

- **One page.** The new workspace starts with a single `PLAN.md`: a goal, the milestone ladder, and the current milestone's checklist.
- **Just-in-time detail.** Write a milestone's detailed breakdown only when you start it.
- **Every milestone ends with something you can watch:** a video, a plot, or the robot doing something.
- **Decisions get one line each** in a "Decisions" list in `PLAN.md`, with a date. They don't get a doc each.

---

## 8. Milestone ladder (draft)

| # | Milestone | You learn | Done when |
|---|---|---|---|
| M0 | Toolchain on the 3080 PC; run stock examples headless: cartpole, then a stock legged locomotion env | PPO basics; envs / observations / actions / rewards; training loop and logging | A trained policy plays back as a video you rendered |
| M1 | Hand-written two-wheel inverted pendulum in MJCF; train balance + velocity tracking | MJCF modeling, reward shaping, domain randomization, latency modeling | Balances and tracks velocity under randomized mass, friction, and latency |
| M2 | Konbini v0: simplified geometry, leg via linkage lookup tables or equality constraints, neck + head, **no tail**. Command-conditioned policy. Draft the command / reference interface | URDF→MJCF conversion, scaling up, command conditioning | Balances while following leg-height and neck commands |
| M3 | Deployment path: ONNX export, a ROS 2 policy controller in `konbinibot`, and a sim-to-sim check (policy through ROS against a second simulator) | The bridge back to hardware | Policy runs through the ROS stack in sim |
| M4 | Actuator models from bench data; first tethered hardware test (needs the hardware track) | Sim2real, actuator system identification | Robot balances on a tether |
| M5 | Tail model + tail in the policy | Continuum modeling in a rigid-body sim | Tail adds measurable balance authority in sim; validated against `tail_eom_*` |
| M6 | Behavior layer: reference / animation engine + reference-tracking policy (BD-X); map §9 primitives (APPROACH, ATTEND, INSPECT, …) | The ELEGNT layer | Expressive primitives play out on the robot |

**The hardware track runs in parallel in `konbinibot`.** The pieces that gate M4 are:
- hip bench tests
- the hip xacro fix (ISSUES I-5)
- `hardware.launch.py` (I-6)
- a wheel-odometry source (I-4)
- a safe-mode fallback (I-7)

---

## 9. M0 checklist

- [ ] On the 3080 PC: confirm Ubuntu version, NVIDIA driver, CUDA, and free disk (aim for ≥ 50–100 GB).
- [ ] Set up SSH access, so training can run headless and be checked remotely.
- [ ] Create the new repo (name TBD, e.g. `konbini-rl`) containing `PLAN.md` (seeded from §7–§8 of this doc), `README.md`, and an empty layout. Keep the layout minimal until M1 needs more.
- [ ] Set up the Python environment (`uv` or conda) and install MuJoCo + MuJoCo Playground with GPU JAX. Verify that JAX sees the GPU.
- [ ] Run a stock cartpole training headless, then render a rollout video.
- [ ] Run a stock legged-locomotion environment from Playground headless and render a video.
- [ ] Set up a training dashboard (TensorBoard or Weights & Biases).
- [ ] Read Spinning Up's PPO page and Rudin et al. (below) alongside these runs.
- [ ] Write down in `PLAN.md` → Decisions: simulator confirmed (or switch), environment manager, logging tool.

---

## 10. Learning resources (in order)

1. **OpenAI Spinning Up.** RL fundamentals and PPO.
2. **MuJoCo documentation, MJCF modeling chapter.** Bodies, joints, actuators, equality constraints, tendons, sensors.
3. **MuJoCo Playground tutorials / notebooks.**
4. **Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"** (legged_gym). The core recipe.
5. **Grandia et al., BD-X**, Disney Research (`refs/bd_x_paper.md`). The architecture for M6.
6. **Hu et al., ELEGNT**, Apple (`refs/ELEGNT_PAPER.MD`). The behavior framing.
7. *For M4:* **Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots".** Actuator networks and sim2real.
8. *Later / optional:* adversarial motion priors (style rewards learned from animation clips), if explicit references aren't enough for "character".

---

## 11. Open questions (answer as they become relevant)

1. **What should the policy own?** Balance + velocity only, or also leg height, neck, stand-up, jump (the RC plan reserves a "jump" mode)? Head under the policy or separate?
2. **Does the tail exist physically?** A prototype segment is implied by `tail_params.m` ("R = 75 mm for current prototype"). It's needed to measure EI_eff and b_eff.
3. **What is γ concretely?** Options: how far the reference deviates from the functional path, or a blend between functional and expressive clips.
4. **The command / reference interface** (§2). Draft at M2.
5. **Policy rate and action type per actuator:** wheel torque vs. velocity; hip torque vs. position target; neck and head position targets.
6. **Pitch sign convention.** Define it once in the new model.
7. **Onboard compute.** Is the LattePanda sufficient long term if vision-based policies are ever wanted?
