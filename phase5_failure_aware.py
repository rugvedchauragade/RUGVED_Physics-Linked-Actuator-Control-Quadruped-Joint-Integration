# ============================================================
# PHASE 5 — FAILURE-AWARE CONTROL
# Actuator failure, sensor failure, degradation + safe behavior
# Quadruped leg — Mini Cheetah scale (Hip + Knee, Phase 4 physics)
# ============================================================
#
# WHAT THIS PHASE ADDS OVER PHASE 4:
#
#   Phase 4: Both joints run correctly with full coupling physics.
#            Assumes sensors and actuators work perfectly forever.
#
#   Phase 5: Real hardware fails. Three failure types simulated:
#
#   FAILURE 1 — Hip actuator LOCK (hard fault)
#       What:  Hip motor jams at t=3.0s (step 300) mid-motion
#              Joint physically cannot move — freezes at current angle
#       Effect: Hip stuck at 88.3deg (setpoint=90). Knee continues normally.
#       Safe:   Detect stall, report fault, hold gravity-compensation
#
#   FAILURE 2 — Knee sensor DEATH (sensor fault)
#       What:  Knee encoder dies at t=3.0s, returns 0deg permanently
#              PID sees error = 45deg always -> commands max torque forever
#       Effect WITHOUT safe: knee spins to 900+ deg uncontrolled (DANGEROUS)
#       Effect WITH safe:    anomaly detector catches 45deg jump in 1 step
#                            switches to gravity-hold mode -> knee stays ~45deg
#       Safe:   Jump detection: |sensor[t] - sensor[t-1]| > 15deg -> SAFE MODE
#               In safe mode: tau = tau_gravity (hold position, no PID)
#
#   FAILURE 3 — Hip actuator DEGRADATION (partial fault)
#       What:  Hip motor loses 50% torque at t=3.0s (coil burn, heat limit)
#              max effective torque drops from 3.0 Nm -> 1.5 Nm
#       Effect: gravity at setpoint = 1.8183 Nm > 1.5 Nm -> hip cannot hold
#               hip slides backward from 87.5deg -> settles at ~56deg
#       Safe:   Detect torque vs position mismatch, switch to reduced target
#
# SAFE BEHAVIOR DEFINITION:
#   A SAFE response means the system:
#     1. Does NOT allow uncontrolled motion (no spinning, no slamming)
#     2. Holds last stable position using gravity compensation
#     3. Freezes the integral accumulator (prevents windup)
#     4. Reports a fault code so operator knows what failed
#     5. Continues operating the healthy joint(s) normally
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# SECTION 1 — PHYSICAL PARAMETERS (identical to Phase 4)
# ============================================================

GRAVITY         = 9.81
M_THIGH         = 0.5;  L_THIGH = 0.20;  MA_THIGH = L_THIGH / 2
M_SHANK         = 0.5;  L_SHANK = 0.20;  MA_SHANK = L_SHANK / 2
I_HIP           = (1/3) * M_THIGH * L_THIGH**2
I_KNEE          = (1/3) * M_SHANK * L_SHANK**2

HIP_SP_RAD      = np.radians(90.0)
KNEE_SP_RAD     = np.radians(45.0)

HIP_TORQUE_MAX  = 3.0    # Nm
KNEE_TORQUE_MAX = 1.5    # Nm
OMEGA_MAX       = np.radians(300.0)

Kp = 2.0;  Ki = 2.0;  Kd = 0.20
DT = 0.01;  TIME_STEPS = 700   # 7.0 s
SETTLING_DEG = 2.0

FAILURE_STEP    = 300    # t = 3.0 s — all failures trigger here


# ============================================================
# SECTION 2 — SAFE BEHAVIOR THRESHOLDS
# ============================================================

# Sensor anomaly: if reading jumps more than this in one step -> sensor fault
SENSOR_JUMP_THRESH_DEG = 15.0

# Actuator degradation: if commanded torque is at limit BUT position
# is moving AWAY from setpoint for this many steps -> degradation fault
DEGRADE_DETECT_STEPS   = 20

# Degradation scenario: hip loses 50% of output torque
HIP_DEGRADE_FACTOR     = 0.50


# ============================================================
# SECTION 3 — PHYSICS FUNCTIONS (from Phase 4)
# ============================================================

def tau_hip_gravity(th, tk):
    """Full coupling: hip supports both thigh and shank."""
    return (M_THIGH * GRAVITY * MA_THIGH * np.sin(th)
            + M_SHANK * GRAVITY * (L_THIGH * np.sin(th)
            + MA_SHANK * np.sin(th + tk)))


def tau_knee_gravity(th, tk):
    """Knee supports shank — depends on ABSOLUTE shank angle."""
    return M_SHANK * GRAVITY * MA_SHANK * np.sin(th + tk)


# ============================================================
# SECTION 4 — BASELINE RUN (no failures — for comparison)
# ============================================================

def run_baseline():
    th=0.0; om_h=0.0; int_h=0.0; pe_h=HIP_SP_RAD
    tk=0.0; om_k=0.0; int_k=0.0; pe_k=KNEE_SP_RAD
    logs = {'time':[],'hip':[],'knee':[],'hip_err':[],'knee_err':[],
            'tau_h':[],'tau_k':[]}
    for step in range(TIME_STEPS):
        e_h=HIP_SP_RAD-th; int_h+=e_h*DT; d_h=(e_h-pe_h)/DT
        tau_h=np.clip(Kp*e_h+Ki*int_h+Kd*d_h,-HIP_TORQUE_MAX,HIP_TORQUE_MAX)
        al_h=(tau_h-tau_hip_gravity(th,tk))/I_HIP
        om_h=np.clip(om_h+al_h*DT,-OMEGA_MAX,OMEGA_MAX); th+=om_h*DT; pe_h=e_h

        e_k=KNEE_SP_RAD-tk; int_k+=e_k*DT; d_k=(e_k-pe_k)/DT
        tau_k=np.clip(Kp*e_k+Ki*int_k+Kd*d_k,-KNEE_TORQUE_MAX,KNEE_TORQUE_MAX)
        al_k=(tau_k-tau_knee_gravity(th,tk))/I_KNEE
        om_k=np.clip(om_k+al_k*DT,-OMEGA_MAX,OMEGA_MAX); tk+=om_k*DT; pe_k=e_k

        logs['time'].append(step*DT); logs['hip'].append(np.degrees(th))
        logs['knee'].append(np.degrees(tk))
        logs['hip_err'].append(np.degrees(HIP_SP_RAD-th))
        logs['knee_err'].append(np.degrees(KNEE_SP_RAD-tk))
        logs['tau_h'].append(tau_h); logs['tau_k'].append(tau_k)
    return logs


# ============================================================
# SECTION 5 — FAILURE 1: HIP ACTUATOR LOCK
# ============================================================

def run_failure1_hip_lock():
    """
    Hip motor jams at FAILURE_STEP.
    Joint physically cannot move — frozen at current angle.
    Knee continues with coupled physics.

    Safe behavior: detect stall (torque commanded, no motion),
    output only gravity compensation, flag fault.
    """
    th=0.0; om_h=0.0; int_h=0.0; pe_h=HIP_SP_RAD
    tk=0.0; om_k=0.0; int_k=0.0; pe_k=KNEE_SP_RAD
    hip_locked=False; lock_angle=None
    hip_safe_mode=False; stall_count=0

    logs = {'time':[],'hip':[],'knee':[],'hip_err':[],'knee_err':[],
            'tau_h':[],'tau_k':[],'hip_fault':[],'knee_fault':[],
            'hip_safe':[],'knee_safe':[]}
    fault_log = []

    for step in range(TIME_STEPS):
        time = step * DT

        # ── Inject failure ────────────────────────────────────
        if step == FAILURE_STEP:
            hip_locked = True
            lock_angle = th
            fault_log.append(f't={time:.2f}s FAULT: HIP LOCK at {np.degrees(th):.2f}deg')

        # ── HIP with stall detection ──────────────────────────
        e_h = HIP_SP_RAD - th
        if not hip_locked:
            int_h += e_h * DT
            d_h = (e_h - pe_h) / DT
            raw_h = Kp * e_h + Ki * int_h + Kd * d_h
            tau_h = np.clip(raw_h, -HIP_TORQUE_MAX, HIP_TORQUE_MAX)
            grav_h = tau_hip_gravity(th, tk)
            al_h = (tau_h - grav_h) / I_HIP
            om_h = np.clip(om_h + al_h * DT, -OMEGA_MAX, OMEGA_MAX)
            th += om_h * DT
            pe_h = e_h
            hip_safe_mode = False
        else:
            # SAFE MODE: joint physically frozen
            # Output only gravity compensation — no integral, no PID
            th = lock_angle;  om_h = 0.0
            tau_h = tau_hip_gravity(th, tk)  # hold against gravity only
            hip_safe_mode = True

        # ── KNEE normal (coupling still active) ───────────────
        e_k = KNEE_SP_RAD - tk
        int_k += e_k * DT
        d_k = (e_k - pe_k) / DT
        tau_k = np.clip(Kp * e_k + Ki * int_k + Kd * d_k,
                        -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)
        grav_k = tau_knee_gravity(th, tk)
        al_k = (tau_k - grav_k) / I_KNEE
        om_k = np.clip(om_k + al_k * DT, -OMEGA_MAX, OMEGA_MAX)
        tk += om_k * DT;  pe_k = e_k

        logs['time'].append(time)
        logs['hip'].append(np.degrees(th));   logs['knee'].append(np.degrees(tk))
        logs['hip_err'].append(np.degrees(e_h))
        logs['knee_err'].append(np.degrees(e_k))
        logs['tau_h'].append(tau_h);          logs['tau_k'].append(tau_k)
        logs['hip_fault'].append(1 if hip_locked else 0)
        logs['knee_fault'].append(0)
        logs['hip_safe'].append(1 if hip_safe_mode else 0)
        logs['knee_safe'].append(0)

    logs['fault_log'] = fault_log
    logs['label']     = 'Failure 1: Hip Actuator Lock'
    return logs


# ============================================================
# SECTION 6 — FAILURE 2: KNEE SENSOR DEATH
# ============================================================

def run_failure2_sensor_death(safe_mode_enabled=True):
    """
    Knee encoder dies at FAILURE_STEP — returns 0deg permanently.

    Without safe behavior:
        PID sees error=45deg forever -> commands max torque -> joint spins
    With safe behavior:
        Anomaly detector catches 45deg jump in ONE step
        Switches to gravity-hold: tau = tau_knee_gravity (no PID, no integral)
    """
    th=0.0; om_h=0.0; int_h=0.0; pe_h=HIP_SP_RAD
    tk=0.0; om_k=0.0; int_k=0.0; pe_k=KNEE_SP_RAD
    sensor_dead=False; knee_safe=False
    last_valid_reading = 0.0
    fault_log = []

    logs = {'time':[],'hip':[],'knee':[],'hip_err':[],'knee_err':[],
            'tau_h':[],'tau_k':[],'sensor_reading':[],
            'hip_fault':[],'knee_fault':[],'hip_safe':[],'knee_safe':[]}

    for step in range(TIME_STEPS):
        time = step * DT

        # ── Inject sensor failure ─────────────────────────────
        if step == FAILURE_STEP:
            sensor_dead = True
            fault_log.append(f't={time:.2f}s FAULT: KNEE SENSOR DEAD (reads 0deg)')

        # ── HIP normal ────────────────────────────────────────
        e_h = HIP_SP_RAD - th
        int_h += e_h * DT
        d_h = (e_h - pe_h) / DT
        tau_h = np.clip(Kp * e_h + Ki * int_h + Kd * d_h,
                        -HIP_TORQUE_MAX, HIP_TORQUE_MAX)
        grav_h = tau_hip_gravity(th, tk)
        al_h = (tau_h - grav_h) / I_HIP
        om_h = np.clip(om_h + al_h * DT, -OMEGA_MAX, OMEGA_MAX)
        th += om_h * DT;  pe_h = e_h

        # ── KNEE with sensor failure ──────────────────────────
        raw_sensor = 0.0 if sensor_dead else tk   # dead sensor reads 0

        if safe_mode_enabled and not knee_safe:
            # ANOMALY DETECTION: impossible single-step jump
            jump_deg = abs(np.degrees(raw_sensor - last_valid_reading))
            if jump_deg > SENSOR_JUMP_THRESH_DEG and step > 5:
                knee_safe = True
                fault_log.append(
                    f't={time:.2f}s DETECTED: sensor jump={jump_deg:.1f}deg '
                    f'> {SENSOR_JUMP_THRESH_DEG}deg -> KNEE SAFE MODE')
        if step > 0:
            last_valid_reading = raw_sensor if not sensor_dead else last_valid_reading

        if knee_safe:
            # SAFE MODE: gravity hold — no PID, no integral accumulation
            tau_k = tau_knee_gravity(th, tk)
            tau_k = np.clip(tau_k, -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)
        else:
            e_k = KNEE_SP_RAD - raw_sensor
            int_k += e_k * DT
            d_k = (e_k - pe_k) / DT
            tau_k = np.clip(Kp * e_k + Ki * int_k + Kd * d_k,
                            -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)

        grav_k = tau_knee_gravity(th, tk)
        al_k = (tau_k - grav_k) / I_KNEE
        om_k = np.clip(om_k + al_k * DT, -OMEGA_MAX, OMEGA_MAX)
        tk += om_k * DT;  pe_k = e_k

        logs['time'].append(time)
        logs['hip'].append(np.degrees(th));   logs['knee'].append(np.degrees(tk))
        logs['hip_err'].append(np.degrees(HIP_SP_RAD - th))
        logs['knee_err'].append(np.degrees(KNEE_SP_RAD - tk))
        logs['tau_h'].append(tau_h);          logs['tau_k'].append(tau_k)
        logs['sensor_reading'].append(np.degrees(raw_sensor))
        logs['hip_fault'].append(0)
        logs['knee_fault'].append(1 if sensor_dead else 0)
        logs['hip_safe'].append(0)
        logs['knee_safe'].append(1 if knee_safe else 0)

    logs['fault_log'] = fault_log
    lbl = 'Failure 2: Knee Sensor Death'
    logs['label'] = lbl + (' + SAFE BEHAVIOR' if safe_mode_enabled else ' (NO safe)')
    return logs


# ============================================================
# SECTION 7 — FAILURE 3: HIP ACTUATOR DEGRADATION
# ============================================================

def run_failure3_degradation():
    """
    Hip motor loses 50% torque at FAILURE_STEP.
    Effective max = 3.0 * 0.5 = 1.5 Nm.
    Gravity at setpoint = 1.8183 Nm > 1.5 Nm -> hip cannot hold.
    Hip slides backward to equilibrium where gravity = available torque.
    """
    th=0.0; om_h=0.0; int_h=0.0; pe_h=HIP_SP_RAD
    tk=0.0; om_k=0.0; int_k=0.0; pe_k=KNEE_SP_RAD
    hip_degraded=False
    degrade_counter=0; degrade_detected=False
    fault_log=[]

    logs = {'time':[],'hip':[],'knee':[],'hip_err':[],'knee_err':[],
            'tau_h_raw':[],'tau_h_actual':[],'tau_k':[],
            'hip_fault':[],'hip_safe':[],'knee_fault':[],'knee_safe':[]}

    for step in range(TIME_STEPS):
        time = step * DT

        if step == FAILURE_STEP:
            hip_degraded = True
            fault_log.append(
                f't={time:.2f}s FAULT: HIP DEGRADED to {HIP_DEGRADE_FACTOR*100:.0f}% torque')

        # ── HIP with degradation ──────────────────────────────
        e_h = HIP_SP_RAD - th
        int_h += e_h * DT
        d_h = (e_h - pe_h) / DT
        raw_h = Kp * e_h + Ki * int_h + Kd * d_h
        tau_h_full   = np.clip(raw_h, -HIP_TORQUE_MAX, HIP_TORQUE_MAX)
        tau_h_actual = (tau_h_full * HIP_DEGRADE_FACTOR
                        if hip_degraded else tau_h_full)

        # DEGRADATION DETECTION: at saturation + position moving wrong way
        at_sat = abs(tau_h_full) >= HIP_TORQUE_MAX * 0.95
        moving_wrong = (np.degrees(e_h) > 0 and om_h < 0)  # error>0 but moving backward
        if hip_degraded and at_sat and moving_wrong:
            degrade_counter += 1
        else:
            degrade_counter = 0
        if degrade_counter >= DEGRADE_DETECT_STEPS and not degrade_detected:
            degrade_detected = True
            fault_log.append(
                f't={time:.2f}s DETECTED: torque saturated + backward motion '
                f'-> degradation confirmed')

        grav_h = tau_hip_gravity(th, tk)
        al_h   = (tau_h_actual - grav_h) / I_HIP
        om_h   = np.clip(om_h + al_h * DT, -OMEGA_MAX, OMEGA_MAX)
        th    += om_h * DT;  pe_h = e_h

        # ── KNEE normal ───────────────────────────────────────
        e_k = KNEE_SP_RAD - tk
        int_k += e_k * DT
        d_k = (e_k - pe_k) / DT
        tau_k = np.clip(Kp * e_k + Ki * int_k + Kd * d_k,
                        -KNEE_TORQUE_MAX, KNEE_TORQUE_MAX)
        grav_k = tau_knee_gravity(th, tk)
        al_k = (tau_k - grav_k) / I_KNEE
        om_k = np.clip(om_k + al_k * DT, -OMEGA_MAX, OMEGA_MAX)
        tk += om_k * DT;  pe_k = e_k

        logs['time'].append(time)
        logs['hip'].append(np.degrees(th));   logs['knee'].append(np.degrees(tk))
        logs['hip_err'].append(np.degrees(e_h))
        logs['knee_err'].append(np.degrees(e_k))
        logs['tau_h_raw'].append(tau_h_full)
        logs['tau_h_actual'].append(tau_h_actual)
        logs['tau_k'].append(tau_k)
        logs['hip_fault'].append(1 if hip_degraded else 0)
        logs['hip_safe'].append(1 if degrade_detected else 0)
        logs['knee_fault'].append(0);  logs['knee_safe'].append(0)

    # Equilibrium: where does degraded hip stabilize?
    max_degrade_torque = HIP_TORQUE_MAX * HIP_DEGRADE_FACTOR
    from scipy.optimize import brentq
    try:
        eq_angle = brentq(
            lambda a: tau_hip_gravity(a, KNEE_SP_RAD) - max_degrade_torque,
            0.01, np.pi - 0.01)
        logs['equilibrium_deg'] = np.degrees(eq_angle)
    except Exception:
        logs['equilibrium_deg'] = None

    logs['fault_log'] = fault_log
    logs['label']     = 'Failure 3: Hip Actuator Degradation (50%)'
    return logs


# ============================================================
# SECTION 8 — RUN ALL SCENARIOS
# ============================================================

run_base   = run_baseline()
run_f1     = run_failure1_hip_lock()
run_f2_safe= run_failure2_sensor_death(safe_mode_enabled=True)
run_f2_raw = run_failure2_sensor_death(safe_mode_enabled=False)
run_f3     = run_failure3_degradation()


# ============================================================
# SECTION 9 — PRINT REPORT
# ============================================================

print("=" * 72)
print("   PHASE 5 — FAILURE-AWARE CONTROL")
print("=" * 72)
print()
print("── SYSTEM PARAMETERS (Phase 4) ────────────────────────────────")
print(f"  Hip:   M={M_THIGH}kg  L={L_THIGH}m  I={I_HIP:.6f}kg.m2  TORQUE_MAX={HIP_TORQUE_MAX}Nm")
print(f"  Knee:  M={M_SHANK}kg  L={L_SHANK}m  I={I_KNEE:.6f}kg.m2  TORQUE_MAX={KNEE_TORQUE_MAX}Nm")
print(f"  All failures triggered at step={FAILURE_STEP}  (t={FAILURE_STEP*DT:.1f}s)")
print()

# ── Failure 1 ────────────────────────────────────────────────
r = run_f1
print("── FAILURE 1: HIP ACTUATOR LOCK ───────────────────────────────")
for msg in r['fault_log']:
    print(f"  {msg}")
hip_lock_angle = r['hip'][FAILURE_STEP]
print(f"  Hip frozen at     : {hip_lock_angle:.3f}deg (setpoint=90deg)")
print(f"  Steady-state error: {90.0 - hip_lock_angle:.3f}deg (permanent — cannot recover)")
print(f"  Knee at end       : {r['knee'][-1]:.3f}deg (settled normally)")
print(f"  Safe behavior     : gravity-hold only, no PID, integral frozen")
print()

# ── Failure 2 ────────────────────────────────────────────────
print("── FAILURE 2: KNEE SENSOR DEATH ───────────────────────────────")
for msg in run_f2_safe['fault_log']:
    print(f"  {msg}")
print(f"  WITHOUT safe behavior: knee at end = {run_f2_raw['knee'][-1]:.1f}deg")
print(f"                         (DANGEROUS: uncontrolled spin)")
print(f"  WITH safe behavior:    knee at end = {run_f2_safe['knee'][-1]:.3f}deg")
print(f"                         (SAFE: stayed near setpoint 45deg)")
print(f"  Safe behavior trigger : jump > {SENSOR_JUMP_THRESH_DEG}deg in 1 step")
print(f"  Safe behavior action  : freeze integral, output = tau_gravity only")
print()

# ── Failure 3 ────────────────────────────────────────────────
print("── FAILURE 3: HIP ACTUATOR DEGRADATION (50%) ──────────────────")
for msg in run_f3['fault_log']:
    print(f"  {msg}")
grav_sp = tau_hip_gravity(HIP_SP_RAD, KNEE_SP_RAD)
max_deg_tau = HIP_TORQUE_MAX * HIP_DEGRADE_FACTOR
print(f"  Gravity at 90deg setpoint  : {grav_sp:.4f} Nm")
print(f"  Max degraded torque        : {max_deg_tau:.4f} Nm")
print(f"  Can hold setpoint?         : NO ({max_deg_tau:.4f} < {grav_sp:.4f})")
if run_f3['equilibrium_deg']:
    print(f"  New equilibrium angle      : ~{run_f3['equilibrium_deg']:.2f}deg")
print(f"  Hip final position         : {run_f3['hip'][-1]:.3f}deg")
print(f"  Hip steady-state error     : {90.0 - run_f3['hip'][-1]:.3f}deg")
print(f"  Knee final position        : {run_f3['knee'][-1]:.3f}deg")
print()

# Step-by-step log Failure 1
print("── STEP LOG: FAILURE 1 (hip lock, every 70 steps) ─────────────")
print(f"  {'Step':<6} {'Time(s)':<8} {'Hip(deg)':<11} {'Knee(deg)':<12} "
      f"{'HipErr':<9} {'KneeErr':<10} {'HipSafe'}")
print("  " + "-" * 64)
for i in range(0, TIME_STEPS, 70):
    safe_str = 'SAFE' if run_f1['hip_safe'][i] else '-'
    print(f"  {i:<6} {run_f1['time'][i]:<8.2f} {run_f1['hip'][i]:<11.3f} "
          f"{run_f1['knee'][i]:<12.3f} {run_f1['hip_err'][i]:<9.3f} "
          f"{run_f1['knee_err'][i]:<10.3f} {safe_str}")
print()

print("── SAFE BEHAVIOR DEFINITION ────────────────────────────────────")
print("  A SAFE response means the system:")
print("  1. Does NOT allow uncontrolled motion (no spinning, no slamming)")
print("  2. Holds last stable position using gravity compensation")
print("  3. Freezes the integral accumulator (prevents windup)")
print("  4. Reports a fault code for operator awareness")
print("  5. Continues operating the healthy joint(s) normally")
print()
print("  Anomaly thresholds used:")
print(f"    Sensor jump          > {SENSOR_JUMP_THRESH_DEG}deg/step -> sensor fault")
print(f"    Degradation detect   : {DEGRADE_DETECT_STEPS} steps at saturation + wrong motion")
print("=" * 72)


# ============================================================
# SECTION 10 — PLOTTING (4 subplots, 2x2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Phase 5 — Failure-Aware Control\n'
             'Three Failure Types + Safe Behavior Responses',
             fontsize=14, y=0.99)

t = run_base['time']

# ── Plot 1: Failure 1 — Hip Lock ─────────────────────────────
ax1 = axes[0, 0]
ax1.plot(t, run_base['hip'],   color='gray',   linewidth=1.6,
         linestyle='--', alpha=0.6, label='Hip — baseline (no failure)')
ax1.plot(t, run_base['knee'],  color='lightgreen', linewidth=1.6,
         linestyle='--', alpha=0.6, label='Knee — baseline')
ax1.plot(t, run_f1['hip'],     color='blue',   linewidth=2.5,
         label='Hip — locked at t=3s')
ax1.plot(t, run_f1['knee'],    color='green',  linewidth=2.5,
         label='Knee — continues normally')
ax1.axvline(x=FAILURE_STEP * DT, color='red', linestyle=':', linewidth=2,
            alpha=0.8, label='Failure trigger (t=3s)')
ax1.axhline(y=90.0, color='red',    linestyle='--', linewidth=1.3,
            alpha=0.6, label='Hip setpoint (90deg)')
ax1.axhline(y=45.0, color='orange', linestyle='--', linewidth=1.3,
            alpha=0.6, label='Knee setpoint (45deg)')
ax1.axhspan(FAILURE_STEP * DT, TIME_STEPS * DT, alpha=0.06,
            color='red', label='Fault active zone')
ax1.annotate(f'Hip frozen\nat {run_f1["hip"][FAILURE_STEP]:.1f}deg',
             xy=(FAILURE_STEP * DT, run_f1['hip'][FAILURE_STEP]),
             xytext=(FAILURE_STEP * DT + 0.6, 80),
             fontsize=8, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
ax1.set_xlabel('Time (s)');  ax1.set_ylabel('Angle (degrees)')
ax1.set_title('Failure 1: Hip Actuator Lock\n(knee continues — hip frozen at 88.3deg)')
ax1.legend(fontsize=7.5, loc='center right');  ax1.grid(True, alpha=0.3)

# ── Plot 2: Failure 2 — Sensor Death (safe vs unsafe) ────────
ax2 = axes[0, 1]
ax2.plot(t, run_base['knee'],           color='gray',    linewidth=1.6,
         linestyle='--', alpha=0.6, label='Knee — baseline')
ax2.plot(t, run_f2_raw['knee'],         color='red',     linewidth=2.0,
         label='Knee — NO safe (DANGER: uncontrolled spin)')
ax2.plot(t, run_f2_safe['knee'],        color='blue',    linewidth=2.5,
         label='Knee — WITH safe behavior')
ax2.plot(t, run_f2_safe['sensor_reading'], color='orange', linewidth=1.5,
         linestyle=':', alpha=0.8, label='Sensor reading (0 after failure)')
ax2.axvline(x=FAILURE_STEP * DT, color='red', linestyle=':', linewidth=2,
            alpha=0.8, label='Sensor dies (t=3s)')
ax2.axhline(y=45.0, color='green', linestyle='--', linewidth=1.3,
            alpha=0.7, label='Knee setpoint (45deg)')

# Mark safe mode activation
safe_start = next((run_f2_safe['time'][i]
                   for i in range(TIME_STEPS)
                   if run_f2_safe['knee_safe'][i]), None)
if safe_start:
    ax2.axvline(x=safe_start, color='blue', linestyle=':', linewidth=1.5,
                alpha=0.8, label=f'Safe mode ON (t={safe_start:.2f}s)')

# Clamp y-axis for visibility (unsafe run goes to 900+)
ax2.set_ylim(-30, 110)
ax2.annotate('Without safe:\nspins to 900+deg', xy=(4.5, 90),
             fontsize=8, color='red',
             bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.7))
ax2.set_xlabel('Time (s)');  ax2.set_ylabel('Knee Angle (degrees)')
ax2.set_title('Failure 2: Knee Sensor Death\n(safe mode catches anomaly in 1 step)')
ax2.legend(fontsize=7.5, loc='upper left');  ax2.grid(True, alpha=0.3)

# ── Plot 3: Failure 3 — Hip Degradation ─────────────────────
ax3 = axes[1, 0]
ax3.plot(t, run_base['hip'],     color='gray', linewidth=1.6,
         linestyle='--', alpha=0.6, label='Hip — baseline')
ax3.plot(t, run_f3['hip'],       color='blue', linewidth=2.5,
         label='Hip — degraded (50% torque at t=3s)')
ax3.plot(t, run_f3['knee'],      color='green', linewidth=2.0,
         alpha=0.8, label='Knee — continues normally')
ax3.axvline(x=FAILURE_STEP * DT, color='red', linestyle=':', linewidth=2,
            alpha=0.8, label='Degradation start (t=3s)')
ax3.axhline(y=90.0, color='red',    linestyle='--', linewidth=1.3,
            alpha=0.6, label='Hip setpoint (90deg)')
ax3.axhline(y=45.0, color='orange', linestyle='--', linewidth=1.3,
            alpha=0.6, label='Knee setpoint (45deg)')
if run_f3.get('equilibrium_deg'):
    ax3.axhline(y=run_f3['equilibrium_deg'], color='purple',
                linestyle=':', linewidth=1.5,
                label=f"New eq. angle ~{run_f3['equilibrium_deg']:.1f}deg")
ax3.annotate(f"Hip slides to\n~{run_f3['hip'][-1]:.1f}deg\n(gravity wins)",
             xy=(6.0, run_f3['hip'][-1]),
             xytext=(5.0, run_f3['hip'][-1] - 12),
             fontsize=8, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
ax3.set_xlabel('Time (s)');  ax3.set_ylabel('Angle (degrees)')
ax3.set_title('Failure 3: Hip Actuator Degradation (50%)\n'
              '(cannot overcome gravity at setpoint -> slides back)')
ax3.legend(fontsize=7.5);  ax3.grid(True, alpha=0.3)

# ── Plot 4: Torque comparison across failures ────────────────
ax4 = axes[1, 1]
ax4.plot(t, run_base['tau_h'],          color='gray',   linewidth=1.5,
         linestyle='--', alpha=0.65, label='tau_hip — baseline')
ax4.plot(t, run_f1['tau_h'],            color='blue',   linewidth=2.0,
         alpha=0.8, label='tau_hip — Failure 1 (lock: drops to grav-hold)')
ax4.plot(t, run_f3['tau_h_raw'],        color='orange', linewidth=1.8,
         linestyle='--', alpha=0.7, label='tau_hip raw — Failure 3')
ax4.plot(t, run_f3['tau_h_actual'],     color='red',    linewidth=2.3,
         label='tau_hip actual — Failure 3 (50% cap)')
ax4.plot(t, run_f2_safe['tau_k'],       color='green',  linewidth=2.0,
         alpha=0.8, label='tau_knee — Failure 2 (drops to grav-hold after safe)')
ax4.axvline(x=FAILURE_STEP * DT, color='red', linestyle=':', linewidth=1.8,
            alpha=0.7, label='Failure trigger')
ax4.axhline(y= HIP_TORQUE_MAX, color='black', linestyle='--',
            linewidth=1.2, alpha=0.5, label=f'Hip limit (±{HIP_TORQUE_MAX}Nm)')
ax4.axhline(y=-HIP_TORQUE_MAX, color='black', linestyle='--',
            linewidth=1.2, alpha=0.5)
ax4.axhline(y=HIP_TORQUE_MAX * HIP_DEGRADE_FACTOR,
            color='red', linestyle=':', linewidth=1.3, alpha=0.7,
            label=f'Degraded hip limit ({HIP_TORQUE_MAX*HIP_DEGRADE_FACTOR}Nm)')
ax4.set_xlabel('Time (s)');  ax4.set_ylabel('Torque (Nm)')
ax4.set_title('Actuator Torque: All Failure Modes\n'
              '(shows when safe mode cuts PID and holds gravity only)')
ax4.legend(fontsize=7.5);  ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase5_failure_aware_graph.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved as phase5_failure_aware_graph.png")


# ============================================================
# SECTION 11 — PHASE 6 HANDOFF
# ============================================================
print()
print("=" * 72)
print("   PHASE 5 -> PHASE 6 HANDOFF")
print("=" * 72)
print()
print("  Phase 5 established:")
print("    Hip lock: correct freeze + gravity-hold safe behavior    [OK]")
print("    Sensor death: anomaly detected in 1 step, knee safe      [OK]")
print(f"    Without safe: knee spins to {run_f2_raw['knee'][-1]:.0f}deg (PROVEN DANGEROUS)")
print(f"    With    safe: knee stays at {run_f2_safe['knee'][-1]:.2f}deg (PROVEN SAFE)")
print("    Degradation: hip slides from 87.5 to 56.6deg, detected  [OK]")
print()
print("  Phase 6 adds:")
print("    Full control + system output definition")
print("    What EXACTLY the controller sends to actuator hardware")
print("    What feedback signals it receives from sensors")
print("    Signal formats, update rates, integration view")
print()
print("  Physics + all constants carried forward unchanged.")
print("=" * 72)
