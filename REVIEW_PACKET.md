# REVIEW_PACKET.md
## Task 2 — Physics-Linked Actuator Control + Quadruped Joint Integration
**Author:** Rugved  
**Task:** Robotics Systems – Task 2  
**Builds on:** Task 1 (PID simulation) → adds physics, two joints, signal chain, failure handling

---

## ENTRY POINT

Run phases in order. Each phase builds on the previous.

```
python phase1_physical_mapping.py      # Define physics (torque, MOI)
python phase2_actuator_pid.py          # PID with real torque units
python phase3_load_aware.py            # Walking gait load switching
python phase4_multijoint.py            # Hip + knee with coupling
python phase5_failure_aware.py         # Failure + safe behavior
python phase6_system_output.py         # Full signal chain (PWM, encoder)
python phase7_final_graphs.py          # Final graphs + analysis
```

**Final output entry point:** `phase7_final_graphs.py`  
Produces all three required graphs and the complete analysis report.

---

## CONTROL FLOW

```
┌──────────────────────────────────────────────────────────────┐
│  CONTROLLER  (100 Hz, DT = 0.01s)                            │
│                                                              │
│  error_rad = setpoint_rad - sensed_angle_rad                 │
│  tau_cmd   = Kp*error + Ki*integral + Kd*derivative   [Nm]  │
│  tau_cmd   = clip(tau_cmd, -TORQUE_MAX, +TORQUE_MAX)         │
└──────────────┬───────────────────────────────────────────────┘
               │ tau_cmd (Nm)
               ▼
┌──────────────────────────────────────────────────────────────┐
│  SIGNAL CONVERSION  (motor driver)                           │
│                                                              │
│  tau_motor  = tau_cmd / GEAR_RATIO          (Nm)            │
│  I_cmd      = tau_motor / Kt                (A)             │
│  V_cmd      = I_cmd * R_winding             (V)             │
│  PWM_duty   = V_cmd / V_supply              [0, 1]          │
└──────────────┬───────────────────────────────────────────────┘
               │ PWM → motor coils
               ▼
┌──────────────────────────────────────────────────────────────┐
│  ACTUATOR PHYSICS  (equation of motion)                      │
│                                                              │
│  tau_gravity = (m_thigh + m_load) * g * (L/2) * sin(theta)  │
│  alpha       = (tau_cmd - tau_gravity) / I       (rad/s²)   │
│  omega      += alpha * DT                        (rad/s)    │
│  theta      += omega * DT                        (rad)      │
└──────────────┬───────────────────────────────────────────────┘
               │ real joint angle
               ▼
┌──────────────────────────────────────────────────────────────┐
│  SENSOR  (12-bit encoder + 30ms delay)                       │
│                                                              │
│  ticks       = theta / (2π) * ENC_RES * GEAR_RATIO          │
│  ticks_noisy = ticks + noise(±2 ticks)                      │
│  sensed_rad  = ticks_noisy / (ENC_RES * GEAR_RATIO) * 2π    │
│  delayed     = sensed_rad[t - SENSOR_DELAY_STEPS]           │
└──────────────┬───────────────────────────────────────────────┘
               │ sensed_angle_rad → back to controller
               └──────────────────────────────────────────────┘
```

### Two-Joint Coupling
The hip controller uses full coupling physics:
```
tau_hip_gravity(th_hip, th_knee) =
    m_thigh * g * (L/2) * sin(th_hip)
  + m_shank * g * [L_thigh * sin(th_hip) + (L_shank/2) * sin(th_hip + th_knee)]
```
The knee gravity torque depends on the absolute shank angle:
```
tau_knee_gravity(th_hip, th_knee) =
    m_shank * g * (L_shank/2) * sin(th_hip + th_knee)
```
This coupling means hip motion disturbs the knee even when the knee is stationary.

---

## ONE REAL OUTPUT

**Nominal run — full signal chain (Phase 7, `run_nominal()`)**

System: Hip setpoint = 90°, Knee setpoint = 45°, both starting at 0°.

```
Hip  final position  : 89.9984 deg   (error = 0.0016 deg)
Knee final position  : 45.0168 deg   (error = 0.0168 deg)

Hip  settling time   : 2.83 s
Knee settling time   : 1.17 s

Hip  max overshoot   : 0.0000 deg    (no overshoot)
Knee max overshoot   : 0.0000 deg

Hip  saturated       : 0.4% of simulation   (first ~0.03s only)
Knee saturated       : 0.1% of simulation

Peak hip  torque     : 3.0000 Nm            (= TORQUE_MAX at start)
Peak knee torque     : 1.5000 Nm            (= TORQUE_MAX at start)
Steady-state hip grav: 1.8182 Nm            (I-term holds this at 90°)
Steady-state knee grav: 0.3467 Nm

Peak motor current   : 5.0000 A             (hip, at saturation)
Peak motor voltage   : 2.5000 V             (well within 24V supply)
PWM at steady state  : 0.0625              (hip, at 90° equilibrium)
```

**Why hip gravity = 1.8182 Nm (not 0.4905 Nm like Task 1):**  
Hip supports BOTH thigh and shank. At (hip=90°, knee=45°):
```
tau_thigh = 0.5 * 9.81 * 0.10 * sin(90°)  = 0.4905 Nm
tau_shank = 0.5 * 9.81 * (0.20*sin(90°) + 0.10*sin(135°))  = 1.3277 Nm
tau_total = 1.8182 Nm  ← 3.7x Task 1 single-joint value
```

---

## ONE FAILURE CASE

**Failure: Knee encoder dies at t = 3.0s**  
File: `phase5_failure_aware.py` → `run_failure2_sensor_death()`  
Also shown in: `phase7_final_graphs.py` → Graph 3

**What happens:**
```
t = 3.00s  FAULT:   Knee sensor dies — returns 0° permanently
t = 3.00s  DETECT:  Sensor jump = 45.65° > 15° threshold → SAFE MODE ON
```

**Without safe behavior (no protection):**
```
Sensor reads 0° → PID error = 45° always → commands TORQUE_MAX constantly
Knee joint spins uncontrolled
Knee final position: 1542.6 degrees   ← CATASTROPHIC FAILURE
Hip also destabilizes due to coupling
```

**With safe behavior (jump detection active):**
```
Jump detected in 1 step (10ms) after sensor dies
Integral accumulator frozen immediately
Controller switches to: tau_k = tau_knee_gravity(th, tk)  (gravity hold only)
Knee final position: 44.40 degrees    ← SAFE (within 1° of setpoint)
Hip continues operating normally: 89.96 degrees
```

**Safe behavior definition:**
1. Detect: `|sensor[t] - sensor[t-1]| > 15°` in one step
2. Freeze: stop integral accumulation immediately
3. Hold: output only `tau = tau_gravity` (prevents drift)
4. Continue: healthy joint (hip) keeps running normally
5. Report: fault code logged with timestamp

---

## SYSTEM PARAMETERS SUMMARY

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Thigh mass | 0.5 | kg | Mini Cheetah scale |
| Thigh length | 0.20 | m | |
| Shank mass | 0.5 | kg | |
| Shank length | 0.20 | m | |
| Moment of inertia | 0.006667 | kg·m² | I = (1/3)mL² |
| Hip TORQUE_MAX | 3.0 | Nm | Carries both links |
| Knee TORQUE_MAX | 1.5 | Nm | Shank only |
| OMEGA_MAX | 300 | deg/s | Both joints |
| Kp | 2.0 | Nm/rad | |
| Ki | 2.0 | Nm/(rad·s) | |
| Kd | 0.20 | Nm·s/rad | |
| Control rate | 100 | Hz | DT = 0.01s |
| Motor Kt | 0.10 | Nm/A | Torque constant |
| Winding R | 0.50 | Ω | |
| Supply V | 24.0 | V | Battery |
| Gear ratio | 6:1 | — | |
| Encoder | 4096 | ticks/rev | 12-bit absolute |
| Sensor delay | 30 | ms | 3 steps SPI latency |
| Encoder noise | ±2 | ticks | ≈ ±0.044° |

---

## PHASE SUMMARY

| Phase | File | What it adds | Key result |
|-------|------|-------------|------------|
| 1 | phase1_physical_mapping.py | Angle → torque mapping, MOI | tau_grav at 90° = 0.4905 Nm |
| 2 | phase2_actuator_pid.py | PID in Nm, torque saturation | Error < 0.001° at SS |
| 3 | phase3_load_aware.py | Walking gait load switching | ±7.15° error from I-term lag |
| 4 | phase4_multijoint.py | Hip + knee coupling | Hip 3.7× heavier, settles@2.9s |
| 5 | phase5_failure_aware.py | 3 failure types + safe mode | 1542° → 44° with safe mode |
| 6 | phase6_system_output.py | PWM, current, encoder ticks | Full hardware signal chain |
| 7 | phase7_final_graphs.py | Final 3 graphs + analysis | All results combined |

---

## LIMITATIONS

- Single leg only (4 legs needed for full quadruped gait)
- No friction model in joint dynamics
- No back-EMF term in motor model (V ≈ I × R only)
- Walking load model is simplified (actual stance force varies with body posture)
- No anti-windup on integral during saturation
- PID gains manually tuned — not optimized via Ziegler-Nichols or LQR
- Simulation only — no real hardware interface implemented

---

## TASK 1 → TASK 2 UPGRADE SUMMARY

| Aspect | Task 1 | Task 2 |
|--------|--------|--------|
| Position update | `pos += ctrl * DT` | `alpha = (tau-grav)/I; omega; theta` |
| Gravity | Not modeled | `tau_grav = m*g*(L/2)*sin(θ)` in Nm |
| Torque units | Dimensionless | Newton-metres (Nm) |
| Joints | 1 (single hip) | 2 (hip + knee, coupled physics) |
| Actuator model | None | Kt, gear ratio, torque limits |
| Signal output | Abstract number | PWM duty, motor current (A) |
| Sensor model | Direct angle read | 12-bit encoder + noise + 30ms delay |
| Walking load | Not simulated | Stance/swing gait cycle (Phase 3) |
| Failure handling | None | 3 failures + jump-detect safe mode |
| Signal chain | Not defined | tau → PWM → physics → encoder → PID |
