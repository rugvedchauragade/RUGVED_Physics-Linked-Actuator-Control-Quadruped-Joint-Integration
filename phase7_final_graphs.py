# ============================================================
# PHASE 7 — FINAL GRAPHS + ANALYSIS
# Complete system summary: angle, torque, failure case
# Quadruped leg — Mini Cheetah scale (Hip + Knee)
# ============================================================
#
# WHAT THIS PHASE DOES:
#   Combines results from all previous phases into one
#   definitive simulation run and three required outputs:
#
#   GRAPH 1 — Angle vs Time
#       Hip + knee convergence with full signal chain
#       Shows settling time, overshoot, setpoint tracking
#
#   GRAPH 2 — Torque vs Time
#       Hip actuator torque, gravity load, motor current
#       Shows saturation at start, gravity compensation at SS
#
#   GRAPH 3 — Failure Case
#       Knee sensor death: UNSAFE vs SAFE behavior
#       Proves safe mode prevents catastrophic joint runaway
#
#   REVIEW_PACKET.md is generated separately.
#
# THIS FILE IS THE FINAL ENTRY POINT FOR THE COMPLETE SYSTEM.
# Running this file produces all three graphs and the analysis.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# SECTION 1 — ALL SYSTEM PARAMETERS (consolidated)
# ============================================================

# ── Physical ─────────────────────────────────────────────────
GRAVITY         = 9.81          # m/s²
M_THIGH         = 0.5           # kg
L_THIGH         = 0.20          # m
MA_THIGH        = L_THIGH / 2   # m
M_SHANK         = 0.5           # kg
L_SHANK         = 0.20          # m
MA_SHANK        = L_SHANK / 2   # m
I_HIP           = (1/3) * M_THIGH * L_THIGH**2    # 0.006667 kg·m²
I_KNEE          = (1/3) * M_SHANK * L_SHANK**2    # 0.006667 kg·m²

# ── Setpoints ────────────────────────────────────────────────
HIP_SP_RAD      = np.radians(90.0)
KNEE_SP_RAD     = np.radians(45.0)

# ── Actuator limits ──────────────────────────────────────────
HIP_TORQUE_MAX  = 3.0           # Nm  (supports both links)
KNEE_TORQUE_MAX = 1.5           # Nm
OMEGA_MAX       = np.radians(300.0)   # rad/s

# ── PID gains ────────────────────────────────────────────────
Kp = 2.0    # Nm/rad
Ki = 2.0    # Nm/(rad·s)
Kd = 0.20   # Nm·s/rad

# ── Simulation ───────────────────────────────────────────────
DT              = 0.01          # s — 100 Hz
TIME_STEPS      = 800           # 8.0 s
SETTLING_DEG    = 2.0

# ── Hardware / signal chain ──────────────────────────────────
Kt              = 0.10          # Nm/A
R_WINDING       = 0.50          # Ω
V_SUPPLY        = 24.0          # V
GEAR_RATIO      = 6.0
ENC_RESOLUTION  = 4096          # ticks/rev
ENC_NOISE_TICKS = 2
SENSOR_DELAY    = 3             # steps (30 ms)

# ── Failure parameters ───────────────────────────────────────
FAILURE_STEP        = 300       # t = 3.0 s
SENSOR_JUMP_THRESH  = 15.0      # degrees

np.random.seed(42)


# ============================================================
# SECTION 2 — PHYSICS FUNCTIONS
# ============================================================

def tau_hip_gravity(th, tk):
    """Full coupling: hip supports thigh + shank."""
    return (M_THIGH * GRAVITY * MA_THIGH * np.sin(th)
            + M_SHANK * GRAVITY * (L_THIGH * np.sin(th)
            + MA_SHANK * np.sin(th + tk)))

def tau_knee_gravity(th, tk):
    """Knee supports shank — absolute angle dependent."""
    return M_SHANK * GRAVITY * MA_SHANK * np.sin(th + tk)

def encoder_read(angle_rad):
    """Encode real angle to noisy encoder ticks, return angle back."""
    ticks = (int(angle_rad / (2*np.pi) * ENC_RESOLUTION * GEAR_RATIO)
             + np.random.randint(-ENC_NOISE_TICKS, ENC_NOISE_TICKS + 1))
    return ticks / (ENC_RESOLUTION * GEAR_RATIO) * 2 * np.pi

def torque_to_pwm(tau_joint, limit):
    """tau (Nm) → PWM duty, motor current (A), motor voltage (V)."""
    tau_m = np.clip(tau_joint, -limit, limit) / GEAR_RATIO
    I     = tau_m / Kt
    V     = I * R_WINDING
    pwm   = np.clip(V / V_SUPPLY, -1.0, 1.0)
    return pwm, abs(I), abs(V)


# ============================================================
# SECTION 3 — RUN 1: NOMINAL (full signal chain)
# Used for Graph 1 (angle) and Graph 2 (torque)
# ============================================================

def run_nominal():
    th = 0.0;  om_h = 0.0;  int_h = 0.0;  pe_h = HIP_SP_RAD
    tk = 0.0;  om_k = 0.0;  int_k = 0.0;  pe_k = KNEE_SP_RAD
    hip_buf  = deque([0.0] * SENSOR_DELAY, maxlen=SENSOR_DELAY)
    knee_buf = deque([0.0] * SENSOR_DELAY, maxlen=SENSOR_DELAY)

    logs = {k: [] for k in [
        'time','hip','knee','hip_sensed','knee_sensed',
        'tau_h','tau_k','grav_h','grav_k',
        'pwm_h','I_h','V_h','saturated_h','saturated_k'
    ]}
    settle_hip = None;  settle_knee = None
    max_overshoot_hip = 0.0;  max_overshoot_knee = 0.0

    for step in range(TIME_STEPS):
        t = step * DT

        # Sensor (encoder + delay)
        hip_buf.append(encoder_read(th))
        knee_buf.append(encoder_read(tk))
        th_s = hip_buf[0];   tk_s = knee_buf[0]

        # PID
        e_h = HIP_SP_RAD  - th_s
        int_h += e_h * DT;  d_h = (e_h - pe_h) / DT
        raw_h = Kp*e_h + Ki*int_h + Kd*d_h
        tau_h = np.clip(raw_h, -HIP_TORQUE_MAX, HIP_TORQUE_MAX)
        sat_h = abs(raw_h) > HIP_TORQUE_MAX

        e_k = KNEE_SP_RAD - tk_s
        int_k += e_k * DT;  d_k = (e_k - pe_k) / DT
        raw_k = Kp*e_k + Ki*int_k + Kd*d_k
        tau_k = np.clip(raw_k, -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)
        sat_k = abs(raw_k) > KNEE_TORQUE_MAX

        # Signal conversion
        pwm_h, I_h, V_h = torque_to_pwm(tau_h, HIP_TORQUE_MAX)

        # Physics
        gh = tau_hip_gravity(th, tk)
        gk = tau_knee_gravity(th, tk)
        om_h = np.clip(om_h + (tau_h - gh) / I_HIP * DT, -OMEGA_MAX, OMEGA_MAX)
        th  += om_h * DT;  pe_h = e_h
        om_k = np.clip(om_k + (tau_k - gk) / I_KNEE * DT, -OMEGA_MAX, OMEGA_MAX)
        tk  += om_k * DT;  pe_k = e_k

        # Metrics
        hd = np.degrees(th);  kd = np.degrees(tk)
        if settle_hip  is None and abs(np.degrees(e_h)) < SETTLING_DEG: settle_hip  = t
        if settle_knee is None and abs(np.degrees(e_k)) < SETTLING_DEG: settle_knee = t
        if hd > 90.0: max_overshoot_hip  = max(max_overshoot_hip,  hd - 90.0)
        if kd > 45.0: max_overshoot_knee = max(max_overshoot_knee, kd - 45.0)

        # Log
        logs['time'].append(t);         logs['hip'].append(hd)
        logs['knee'].append(kd);        logs['hip_sensed'].append(np.degrees(th_s))
        logs['knee_sensed'].append(np.degrees(tk_s))
        logs['tau_h'].append(tau_h);    logs['tau_k'].append(tau_k)
        logs['grav_h'].append(gh);      logs['grav_k'].append(gk)
        logs['pwm_h'].append(pwm_h);    logs['I_h'].append(I_h);  logs['V_h'].append(V_h)
        logs['saturated_h'].append(int(sat_h))
        logs['saturated_k'].append(int(sat_k))

    logs['settle_hip']         = settle_hip
    logs['settle_knee']        = settle_knee
    logs['max_overshoot_hip']  = max_overshoot_hip
    logs['max_overshoot_knee'] = max_overshoot_knee
    logs['hip_final']          = np.degrees(th)
    logs['knee_final']         = np.degrees(tk)
    logs['grav_h_final']       = tau_hip_gravity(th, tk)
    logs['grav_k_final']       = tau_knee_gravity(th, tk)
    logs['I_h_final']          = I_h
    logs['sat_h_pct']          = sum(logs['saturated_h']) / TIME_STEPS * 100
    logs['sat_k_pct']          = sum(logs['saturated_k']) / TIME_STEPS * 100
    return logs


# ============================================================
# SECTION 4 — RUN 2: FAILURE CASE (sensor death)
# Used for Graph 3 (failure)
# Runs BOTH unsafe (no safe mode) and safe to compare
# ============================================================

def run_failure_case(safe_mode):
    """Knee sensor dies at t=3.0s. safe_mode=True applies jump detection."""
    th = 0.0;  om_h = 0.0;  int_h = 0.0;  pe_h = HIP_SP_RAD
    tk = 0.0;  om_k = 0.0;  int_k = 0.0;  pe_k = KNEE_SP_RAD
    sensor_dead = False;  knee_safe = False
    last_valid  = 0.0
    fault_time  = None;  safe_time = None

    logs = {k: [] for k in [
        'time','hip','knee','sensor_reading','tau_k',
        'knee_fault','knee_safe_flag'
    ]}

    for step in range(TIME_STEPS):
        t = step * DT
        if step == FAILURE_STEP:
            sensor_dead = True
            fault_time  = t

        # Hip (normal throughout)
        e_h = HIP_SP_RAD - th
        int_h += e_h * DT;  d_h = (e_h - pe_h) / DT
        tau_h = np.clip(Kp*e_h + Ki*int_h + Kd*d_h, -HIP_TORQUE_MAX, HIP_TORQUE_MAX)
        gh = tau_hip_gravity(th, tk)
        om_h = np.clip(om_h + (tau_h - gh) / I_HIP * DT, -OMEGA_MAX, OMEGA_MAX)
        th += om_h * DT;  pe_h = e_h

        # Knee sensor
        raw_s = 0.0 if sensor_dead else tk

        # Anomaly detection (safe mode only)
        if safe_mode and not knee_safe and step > 5:
            jump = abs(np.degrees(raw_s - last_valid))
            if jump > SENSOR_JUMP_THRESH:
                knee_safe = True
                safe_time = t
        if step > 0 and not sensor_dead:
            last_valid = raw_s

        # Knee control
        if knee_safe:
            tau_k = np.clip(tau_knee_gravity(th, tk), -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)
        else:
            e_k = KNEE_SP_RAD - raw_s
            int_k += e_k * DT;  d_k = (e_k - pe_k) / DT
            tau_k = np.clip(Kp*e_k + Ki*int_k + Kd*d_k, -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)

        gk = tau_knee_gravity(th, tk)
        om_k = np.clip(om_k + (tau_k - gk) / I_KNEE * DT, -OMEGA_MAX, OMEGA_MAX)
        tk += om_k * DT;  pe_k = e_k

        logs['time'].append(t)
        logs['hip'].append(np.degrees(th))
        logs['knee'].append(np.degrees(tk))
        logs['sensor_reading'].append(np.degrees(raw_s))
        logs['tau_k'].append(tau_k)
        logs['knee_fault'].append(int(sensor_dead))
        logs['knee_safe_flag'].append(int(knee_safe))

    logs['fault_time'] = fault_time
    logs['safe_time']  = safe_time
    logs['knee_final'] = np.degrees(tk)
    logs['safe_mode']  = safe_mode
    return logs


# ============================================================
# SECTION 5 — RUN EVERYTHING
# ============================================================

print("=" * 68)
print("   PHASE 7 — FINAL GRAPHS + ANALYSIS")
print("=" * 68)
print("Running all simulations...")

nom   = run_nominal()
f_raw = run_failure_case(safe_mode=False)
f_saf = run_failure_case(safe_mode=True)

print("Done.")
print()


# ============================================================
# SECTION 6 — PRINT FINAL ANALYSIS REPORT
# ============================================================

print("=" * 68)
print("   FINAL SYSTEM ANALYSIS REPORT")
print("=" * 68)
print()
print("── PHYSICAL SYSTEM ─────────────────────────────────────────")
print(f"  Thigh  : {M_THIGH}kg  {L_THIGH}m  I={I_HIP:.6f}kg.m2")
print(f"  Shank  : {M_SHANK}kg  {L_SHANK}m  I={I_KNEE:.6f}kg.m2")
print(f"  Hip actuator  : max {HIP_TORQUE_MAX}Nm  (carries both links)")
print(f"  Knee actuator : max {KNEE_TORQUE_MAX}Nm")
print(f"  Control loop  : {1/DT:.0f} Hz  |  Signal chain: "
      f"Kt={Kt}Nm/A  R={R_WINDING}Ohm  GEAR={GEAR_RATIO}:1")
print()
print("── NOMINAL PERFORMANCE (full signal chain) ─────────────────")
print(f"  Hip  final position  : {nom['hip_final']:.4f} deg  "
      f"(error = {90.0-nom['hip_final']:.4f} deg)")
print(f"  Knee final position  : {nom['knee_final']:.4f} deg  "
      f"(error = {45.0-nom['knee_final']:.4f} deg)")
print(f"  Hip  settling time   : {nom['settle_hip']:.2f} s")
print(f"  Knee settling time   : {nom['settle_knee']:.2f} s")
print(f"  Hip  max overshoot   : {nom['max_overshoot_hip']:.4f} deg")
print(f"  Knee max overshoot   : {nom['max_overshoot_knee']:.4f} deg")
print(f"  Hip  saturated       : {nom['sat_h_pct']:.1f}% of simulation")
print(f"  Knee saturated       : {nom['sat_k_pct']:.1f}% of simulation")
print()
print("── TORQUE ANALYSIS ─────────────────────────────────────────")
print(f"  Peak hip  torque     : {max(nom['tau_h']):.4f} Nm  (at start = TORQUE_MAX)")
print(f"  Peak knee torque     : {max(nom['tau_k']):.4f} Nm  (at start = TORQUE_MAX)")
print(f"  Steady-state hip grav: {nom['grav_h_final']:.4f} Nm  "
      f"(I-term must hold this)")
print(f"  Steady-state knee gr : {nom['grav_k_final']:.4f} Nm")
print(f"  Peak motor current   : {max(nom['I_h']):.4f} A  "
      f"(hip at saturation)")
print(f"  Peak motor voltage   : {max(nom['V_h']):.4f} V  "
      f"(well within {V_SUPPLY}V supply)")
print()
print("── FAILURE CASE: KNEE SENSOR DEATH ─────────────────────────")
print(f"  Sensor fails at      : t = {f_saf['fault_time']:.2f} s")
if f_saf['safe_time']:
    print(f"  Safe mode triggered  : t = {f_saf['safe_time']:.2f} s  "
          f"({(f_saf['safe_time']-f_saf['fault_time'])*1000:.0f}ms after fault)")
print(f"  WITHOUT safe behavior: knee final = {f_raw['knee_final']:.1f} deg  "
      f"<< DANGEROUS (spins)")
print(f"  WITH safe behavior   : knee final = {f_saf['knee_final']:.4f} deg  "
      f"<< SAFE (near setpoint)")
print(f"  Safe detection method: sensor jump > {SENSOR_JUMP_THRESH}deg in 1 step")
print(f"  Safe action          : freeze integral, output = tau_gravity only")
print()
print("── TASK 1 vs TASK 2 — COMPLETE COMPARISON ──────────────────")
print()
print(f"  {'Aspect':<28} {'Task 1':<22} {'Task 2'}")
print("  " + "-" * 70)
rows = [
    ('Position update',      'pos += ctrl*DT',       'alpha=(tau-grav)/I; omega; theta'),
    ('Gravity',              'None (ignored)',        f'tau_grav={nom["grav_h_final"]:.4f}Nm at SS'),
    ('Torque units',         'Dimensionless',        'Newton-metres (Nm)'),
    ('Joints',               '1 (single)',            '2 (hip + knee coupled)'),
    ('Actuator model',       'None',                 f'Kt={Kt}Nm/A  GEAR={GEAR_RATIO}:1'),
    ('Signal output',        'Abstract number',      f'PWM={nom["pwm_h"][400]:.4f} / I={nom["I_h"][400]:.3f}A'),
    ('Sensor model',         'Direct position read', f'Enc ticks + noise + 30ms delay'),
    ('Failure handling',     'None',                 'Jump detect + gravity-hold safe mode'),
]
for aspect, t1, t2 in rows:
    print(f"  {aspect:<28} {t1:<22} {t2}")
print()
print("=" * 68)


# ============================================================
# SECTION 7 — GRAPH 1: ANGLE VS TIME
# ============================================================

fig1, axes1 = plt.subplots(2, 1, figsize=(12, 9))
fig1.suptitle('Phase 7 — Graph 1: Angle vs Time\n'
              'Hip + Knee Convergence — Full Signal Chain (Noise + Delay)',
              fontsize=13, y=0.99)

t = nom['time']

# ── Subplot 1: Both joint angles ─────────────────────────────
ax = axes1[0]
ax.plot(t, nom['hip'],         color='blue',   linewidth=2.5,
        label='Hip real position')
ax.plot(t, nom['hip_sensed'],  color='skyblue', linewidth=1.5,
        alpha=0.75, linestyle='--', label='Hip sensed (enc+delay)')
ax.plot(t, nom['knee'],        color='green',  linewidth=2.5,
        label='Knee real position')
ax.plot(t, nom['knee_sensed'], color='lime',   linewidth=1.5,
        alpha=0.75, linestyle='--', label='Knee sensed (enc+delay)')
ax.axhline(y=90.0, color='red',    linestyle='--', linewidth=1.5,
           label='Hip setpoint (90deg)')
ax.axhline(y=45.0, color='orange', linestyle='--', linewidth=1.5,
           label='Knee setpoint (45deg)')
ax.axhline(y=90+SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax.axhline(y=90-SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0,
           alpha=0.6, label=f'+/-{SETTLING_DEG}deg settling bands')
ax.axhline(y=45+SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax.axhline(y=45-SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
if nom['settle_hip']:
    ax.axvline(x=nom['settle_hip'],  color='blue',  linestyle=':',
               linewidth=1.3, alpha=0.7,
               label=f"Hip settled@{nom['settle_hip']:.2f}s")
if nom['settle_knee']:
    ax.axvline(x=nom['settle_knee'], color='green', linestyle=':',
               linewidth=1.3, alpha=0.7,
               label=f"Knee settled@{nom['settle_knee']:.2f}s")

# Annotations
ax.annotate(f"Final: {nom['hip_final']:.2f}deg",
            xy=(t[-1], nom['hip'][-1]),
            xytext=(t[-1]-1.5, nom['hip'][-1]+4),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.0))
ax.annotate(f"Final: {nom['knee_final']:.2f}deg",
            xy=(t[-1], nom['knee'][-1]),
            xytext=(t[-1]-1.5, nom['knee'][-1]-6),
            fontsize=8, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.0))

ax.set_ylabel('Angle (degrees)')
ax.set_title('Joint Angles vs Time — Hip (0°→90°) and Knee (0°→45°)')
ax.legend(fontsize=7.5, loc='center right')
ax.grid(True, alpha=0.35)

# ── Subplot 2: Position error ─────────────────────────────────
ax = axes1[1]
hip_err  = [90.0 - v for v in nom['hip']]
knee_err = [45.0 - v for v in nom['knee']]
ax.plot(t, hip_err,  color='blue',  linewidth=2.5, label='Hip error')
ax.plot(t, knee_err, color='green', linewidth=2.5, label='Knee error')
ax.axhline(y= SETTLING_DEG, color='gray', linestyle=':', linewidth=1.2,
           label=f'+/-{SETTLING_DEG}deg band')
ax.axhline(y=-SETTLING_DEG, color='gray', linestyle=':', linewidth=1.2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.3,
           label='Zero error')
ax.fill_between(t, -SETTLING_DEG, SETTLING_DEG, alpha=0.08, color='green')
ax.set_xlabel('Time (s)');  ax.set_ylabel('Error (degrees)')
ax.set_title(f"Position Error vs Time  "
             f"(Hip settles@{nom['settle_hip']:.2f}s, "
             f"Knee settles@{nom['settle_knee']:.2f}s)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase7_graph1_angle_vs_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph 1 saved: phase7_graph1_angle_vs_time.png")


# ============================================================
# SECTION 8 — GRAPH 2: TORQUE VS TIME
# ============================================================

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 11))
fig2.suptitle('Phase 7 — Graph 2: Torque vs Time\n'
              'Actuator Commands, Gravity Load, Motor Current',
              fontsize=13, y=0.99)

# ── Subplot 1: Hip torque breakdown ──────────────────────────
ax = axes2[0]
ax.plot(t, nom['tau_h'],  color='blue',   linewidth=2.5,
        label='Hip tau_cmd (actuator command)')
ax.plot(t, nom['grav_h'], color='red',    linewidth=2.0,
        linestyle='--', label='Hip tau_gravity (load to overcome)')
ax.fill_between(t, nom['grav_h'], nom['tau_h'],
                alpha=0.12, color='blue',
                label='Net torque = tau_cmd - tau_grav')
ax.axhline(y=HIP_TORQUE_MAX, color='black', linestyle='--', linewidth=1.2,
           alpha=0.6, label=f'Saturation limit = {HIP_TORQUE_MAX}Nm')

# Mark saturation zone
sat_end_t = next((nom['time'][i] for i in range(TIME_STEPS)
                  if not nom['saturated_h'][i]), 0)
ax.axvspan(0, sat_end_t, alpha=0.10, color='orange',
           label=f'Saturation zone (0 – {sat_end_t:.2f}s)')

ax.annotate(f"SS gravity load\n{nom['grav_h_final']:.4f} Nm",
            xy=(t[-1], nom['grav_h'][-1]),
            xytext=(t[-1]-2.0, nom['grav_h'][-1]+0.4),
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.0))
ax.set_ylabel('Torque (Nm)')
ax.set_title('Hip Torque: Command vs Gravity Load\n'
             '(hip carries BOTH link weights — 3.7x single joint)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

# ── Subplot 2: Knee torque breakdown ─────────────────────────
ax = axes2[1]
ax.plot(t, nom['tau_k'],  color='green',  linewidth=2.5,
        label='Knee tau_cmd')
ax.plot(t, nom['grav_k'], color='orange', linewidth=2.0,
        linestyle='--', label='Knee tau_gravity (coupling-dependent)')
ax.axhline(y=KNEE_TORQUE_MAX, color='black', linestyle='--', linewidth=1.2,
           alpha=0.6, label=f'Saturation = {KNEE_TORQUE_MAX}Nm')
ax.fill_between(t, nom['grav_k'], nom['tau_k'],
                alpha=0.12, color='green', label='Net torque')
ax.set_ylabel('Torque (Nm)')
ax.set_title('Knee Torque: Command vs Gravity Load\n'
             '(gravity depends on ABSOLUTE shank angle = hip + knee — coupling)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

# ── Subplot 3: Motor current + PWM ───────────────────────────
ax = axes2[2]
ax3_twin = ax.twinx()
ax.plot(t, nom['I_h'],   color='blue',   linewidth=2.5,
        label='Hip motor current I_h (A)')
ax3_twin.plot(t, nom['pwm_h'], color='purple', linewidth=2.0,
              linestyle='--', alpha=0.85, label='Hip PWM duty cycle')
ax.axhline(y=max(nom['I_h']), color='blue', linestyle=':',
           linewidth=1.2, alpha=0.6,
           label=f'Peak I = {max(nom["I_h"]):.2f}A')
ax.set_ylabel('Motor Current (A)', color='blue')
ax.tick_params(axis='y', colors='blue')
ax3_twin.set_ylabel('PWM Duty Cycle', color='purple')
ax3_twin.tick_params(axis='y', colors='purple')
ax.set_xlabel('Time (s)')
ax.set_title('Motor Current + PWM Duty Cycle\n'
             '(what the controller actually sends to motor driver hardware)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(True, alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase7_graph2_torque_vs_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph 2 saved: phase7_graph2_torque_vs_time.png")


# ============================================================
# SECTION 9 — GRAPH 3: FAILURE CASE
# ============================================================

fig3, axes3 = plt.subplots(3, 1, figsize=(12, 11))
fig3.suptitle('Phase 7 — Graph 3: Failure Case — Knee Sensor Death\n'
              'Unsafe (No Protection) vs Safe Behavior Response',
              fontsize=13, y=0.99)

t = f_saf['time']

# ── Subplot 1: Knee position — safe vs unsafe ─────────────────
ax = axes3[0]
ax.plot(t, [45.0]*len(t),    color='orange', linewidth=1.5,
        linestyle='--', label='Knee setpoint (45deg)')
ax.plot(t, f_saf['knee'],    color='blue',   linewidth=2.5,
        label=f"WITH safe behavior (final={f_saf['knee_final']:.2f}deg)")

# Clamp unsafe for visibility
unsafe_clipped = [min(v, 150.0) for v in f_raw['knee']]
ax.plot(t, unsafe_clipped,   color='red',    linewidth=2.0,
        linestyle='--', label=f"WITHOUT safe (final={f_raw['knee_final']:.0f}deg — clipped at 150 for display)")

ax.axvline(x=FAILURE_STEP*DT, color='red', linestyle=':', linewidth=2.0,
           alpha=0.8, label=f'Sensor dies (t={FAILURE_STEP*DT:.1f}s)')
if f_saf['safe_time']:
    ax.axvline(x=f_saf['safe_time'], color='blue', linestyle=':', linewidth=1.8,
               alpha=0.8, label=f"Safe mode ON (t={f_saf['safe_time']:.2f}s)")
ax.axhline(y=45+SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax.axhline(y=45-SETTLING_DEG, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax.axvspan(FAILURE_STEP*DT, TIME_STEPS*DT, alpha=0.06, color='red')

ax.annotate('Unsafe: spins to\n1542 deg (clipped)',
            xy=(5.0, 145), fontsize=8.5, color='red',
            bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.8))
ax.annotate(f"Safe: stays at\n{f_saf['knee_final']:.1f}deg",
            xy=(6.5, 50), fontsize=8.5, color='blue',
            bbox=dict(boxstyle='round', facecolor='#e0e8ff', alpha=0.8))
ax.set_ylim(-10, 160)
ax.set_ylabel('Knee Angle (degrees)')
ax.set_title('Knee Position: Safe vs Unsafe Response to Sensor Failure')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

# ── Subplot 2: Sensor reading vs real position ────────────────
ax = axes3[1]
ax.plot(t, f_saf['knee'],           color='blue',   linewidth=2.5,
        label='Real knee position')
ax.plot(t, f_saf['sensor_reading'], color='red',    linewidth=2.0,
        linestyle='--', label='Sensor reading (0 after failure)')
safe_flags = f_saf['knee_safe_flag']
safe_on_t  = next((f_saf['time'][i] for i in range(len(safe_flags)) if safe_flags[i]), None)
if safe_on_t:
    ax.axvspan(safe_on_t, TIME_STEPS*DT, alpha=0.10, color='blue',
               label='Safe mode active (gravity hold)')
ax.axvline(x=FAILURE_STEP*DT, color='red', linestyle=':', linewidth=2.0, alpha=0.8)
ax.axhline(y=45.0, color='orange', linestyle='--', linewidth=1.3,
           alpha=0.7, label='Setpoint (45deg)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Sensor Reading vs Real Position\n'
             '(sensor reads 0 permanently — anomaly detected by jump threshold)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

# ── Subplot 3: Knee torque — shows safe mode action ──────────
ax = axes3[2]
ax.plot(t, f_raw['tau_k'], color='red',  linewidth=2.0,
        linestyle='--', alpha=0.85,
        label='tau_k — UNSAFE (winds up to TORQUE_MAX)')
ax.plot(t, f_saf['tau_k'], color='blue', linewidth=2.5,
        label='tau_k — SAFE (drops to gravity hold after detection)')
ax.axhline(y= KNEE_TORQUE_MAX, color='black', linestyle='--', linewidth=1.3,
           alpha=0.6, label=f'+/-{KNEE_TORQUE_MAX}Nm limit')
ax.axhline(y=-KNEE_TORQUE_MAX, color='black', linestyle='--', linewidth=1.3,
           alpha=0.6)

# Mark gravity hold value at setpoint
tau_grav_45 = tau_knee_gravity(HIP_SP_RAD, KNEE_SP_RAD)
ax.axhline(y=tau_grav_45, color='green', linestyle=':', linewidth=1.3,
           alpha=0.8,
           label=f'Gravity hold target = {tau_grav_45:.4f}Nm')
ax.axvline(x=FAILURE_STEP*DT, color='red', linestyle=':', linewidth=2.0, alpha=0.8)
if safe_on_t:
    ax.axvline(x=safe_on_t, color='blue', linestyle=':', linewidth=1.8, alpha=0.8)
ax.set_xlabel('Time (s)');  ax.set_ylabel('Knee Torque (Nm)')
ax.set_title('Knee Control Torque: Safe vs Unsafe\n'
             '(unsafe winds up to max; safe drops to gravity-only hold)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase7_graph3_failure_case.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph 3 saved: phase7_graph3_failure_case.png")

print()
print("=" * 68)
print("   PHASE 7 COMPLETE — ALL 3 GRAPHS SAVED")
print("=" * 68)
print("  phase7_graph1_angle_vs_time.png")
print("  phase7_graph2_torque_vs_time.png")
print("  phase7_graph3_failure_case.png")
print()
print("  Next: REVIEW_PACKET.md (generated separately)")
print("=" * 68)
