# ============================================================
# PHASE 2 — ACTUATOR-CONSTRAINED PID
# Physics-based PID with real torque and speed limits
# Quadruped hip joint — Mini Cheetah scale
# ============================================================
#
# WHAT THIS PHASE ADDS OVER TASK 1:
#
#   Task 1 Phase 2:
#       error → PID → control_output (abstract) → pos += control * DT
#       No units. No physics. Just numbers.
#
#   Task 2 Phase 2:
#       error (rad) → PID → tau_pid (Nm)
#                         → CLAMP to TORQUE_MAX (actuator saturation)
#                         → tau_net = tau_actuator - tau_gravity
#                         → alpha = tau_net / I          (rad/s²)
#                         → omega += alpha * DT          (rad/s)
#                         → CLAMP omega to OMEGA_MAX     (speed limit)
#                         → theta += omega * DT          (radians → degrees)
#
#   PID output is now TORQUE IN NEWTON-METRES — a real physical quantity.
#   The joint moves because of real angular acceleration, not abstract addition.
#
# ACTUATOR MODEL:
#   Small brushless motor + gearbox (Mini Cheetah scale)
#   Max torque : 1.5 Nm  (limited by motor winding and gear ratio)
#   Max speed  : 300 deg/s = 5.236 rad/s
#
# WHAT THIS PHASE DEMONSTRATES:
#   1. Physics-grounded convergence toward setpoint
#   2. Actuator torque saturation at the start of motion
#   3. Gravity compensation handled by the integral term
#   4. Effect of removing the torque cap (comparison run)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1 — PHYSICAL PARAMETERS (from Phase 1)
# ============================================================

JOINT_MASS        = 0.5                       # kg
LINK_LENGTH       = 0.20                      # m
MOMENT_ARM        = LINK_LENGTH / 2           # m   — COM at midpoint
GRAVITY           = 9.81                      # m/s²
MOMENT_OF_INERTIA = (1/3) * JOINT_MASS * LINK_LENGTH**2   # kg·m²

# Setpoint and initial (consistent with Task 1)
SETPOINT_DEG      = 90.0                      # degrees
INITIAL_DEG       = 0.0                       # degrees
SETPOINT_RAD      = np.radians(SETPOINT_DEG)  # radians
INITIAL_RAD       = np.radians(INITIAL_DEG)   # radians

# ============================================================
# SECTION 2 — ACTUATOR CONSTRAINTS
# These come from the real hardware — not made-up numbers
# ============================================================

TORQUE_MAX_NM     = 1.5      # Nm   — peak motor torque (hardware limit)
TORQUE_MIN_NM     = -1.5     # Nm   — bidirectional actuator
OMEGA_MAX_RAD_S   = np.radians(300.0)   # rad/s — 300 deg/s speed limit

# For reference: gravity load at setpoint = 0.4905 Nm
# TORQUE_MAX is 3× the gravity load → controller has headroom
GRAVITY_AT_SETPOINT = JOINT_MASS * GRAVITY * MOMENT_ARM   # sin(90°) = 1

# ============================================================
# SECTION 3 — PID GAINS (tuned in TORQUE units — Nm per rad)
# ============================================================

# Error unit: radians
# PID output unit: Newton-metres (Nm)
#
# Tuning logic:
#   Kp = 2.0 Nm/rad  → at start error=pi/2 rad, P-term = 3.14 Nm
#                       clamped to 1.5 Nm → actuator immediately saturates
#   Ki = 2.0 Nm/(rad·s) → integral builds up to hold against gravity at 90°
#                          needs to accumulate ~0.49 Nm at steady state
#   Kd = 0.20 Nm·s/rad → derivative brakes overshoot as approach setpoint
#
# Verified by simulation:
#   Final error : ~0.000° ✓
#   Overshoot   : ~0.985° ✓ (below 2° settling band)
#   Settled at  : ~0.37s  ✓

Kp = 2.0     # Nm / rad
Ki = 2.0     # Nm / (rad·s)
Kd = 0.20    # Nm·s / rad

# ============================================================
# SECTION 4 — SIMULATION SETTINGS
# ============================================================

DT            = 0.01     # s — 100 Hz control loop (realistic for embedded)
TIME_STEPS    = 500      # steps — 5 seconds total simulation
SETTLING_BAND = 2.0      # degrees — settled when error within ±2°

# ============================================================
# SECTION 5 — PHYSICS FUNCTIONS
# ============================================================

def gravity_torque(theta_rad):
    """
    Resistive torque due to gravity at joint angle theta.
    tau_gravity = m * g * (L/2) * sin(theta)
    Returns Nm.
    """
    return JOINT_MASS * GRAVITY * MOMENT_ARM * np.sin(theta_rad)


def angular_acceleration(tau_actuator, theta_rad):
    """
    Net angular acceleration from actuator torque vs gravity.
    alpha = (tau_actuator - tau_gravity) / I
    Returns rad/s².
    """
    tau_grav = gravity_torque(theta_rad)
    return (tau_actuator - tau_grav) / MOMENT_OF_INERTIA


# ============================================================
# SECTION 6 — SIMULATION RUNNER
# Runs a full PID simulation with optional torque cap
# ============================================================

def run_simulation(torque_cap, label):
    """
    Run physics-based PID simulation.

    Args:
        torque_cap : float — max actuator torque in Nm (use np.inf to remove cap)
        label      : str   — name for this run (used in print output)

    Returns:
        dict with all logged time-series data
    """
    theta     = INITIAL_RAD     # rad — current joint angle
    omega     = 0.0             # rad/s — current angular velocity
    integral  = 0.0             # rad·s — PID integral accumulator
    prev_err  = SETPOINT_RAD - theta   # rad — previous error for derivative

    # Storage
    time_log          = []
    theta_deg_log     = []       # degrees — joint angle
    theta_rad_log     = []       # radians — joint angle
    omega_log         = []       # rad/s — angular velocity
    omega_deg_log     = []       # deg/s — angular velocity
    error_deg_log     = []       # degrees — position error
    tau_pid_log       = []       # Nm — raw PID output
    tau_cmd_log       = []       # Nm — clamped (actual) torque
    tau_gravity_log   = []       # Nm — gravity load at each step
    alpha_log         = []       # rad/s² — angular acceleration
    P_term_log        = []       # Nm — proportional component
    I_term_log        = []       # Nm — integral component
    D_term_log        = []       # Nm — derivative component
    saturated_log     = []       # bool — was actuator saturated?

    settling_time     = None     # s — first time within settling band
    overshoot_deg     = 0.0      # degrees — max above setpoint
    max_omega_deg     = 0.0      # deg/s — peak speed reached

    for step in range(TIME_STEPS):
        time = step * DT

        # ── STEP 1: Calculate error in radians ───────────────
        error_rad  = SETPOINT_RAD - theta
        error_deg  = np.degrees(error_rad)

        # ── STEP 2: PID calculation — output in Nm ───────────
        integral   += error_rad * DT
        derivative  = (error_rad - prev_err) / DT

        P_term = Kp * error_rad
        I_term = Ki * integral
        D_term = Kd * derivative
        tau_pid = P_term + I_term + D_term   # Nm — raw PID output

        # ── STEP 3: Clamp to actuator torque limits ───────────
        # This is the KEY addition vs Task 1 — the actuator has a
        # physical maximum torque it can deliver
        tau_cmd    = np.clip(tau_pid, -torque_cap, torque_cap)
        is_sat     = abs(tau_pid) > torque_cap

        # ── STEP 4: Physics — equation of motion ─────────────
        # alpha = (tau_actuator - tau_gravity) / I
        tau_grav   = gravity_torque(theta)
        alpha      = angular_acceleration(tau_cmd, theta)

        # ── STEP 5: Integrate angular velocity ────────────────
        omega     += alpha * DT

        # ── STEP 6: Clamp angular velocity (speed limit) ──────
        omega      = np.clip(omega, -OMEGA_MAX_RAD_S, OMEGA_MAX_RAD_S)

        # ── STEP 7: Integrate position ────────────────────────
        theta     += omega * DT

        # ── STEP 8: Update previous error ────────────────────
        prev_err   = error_rad

        # ── STEP 9: Stability tracking ────────────────────────
        pos_deg    = np.degrees(theta)
        if pos_deg > SETPOINT_DEG:
            overshoot_deg = max(overshoot_deg, pos_deg - SETPOINT_DEG)
        if settling_time is None and abs(error_deg) < SETTLING_BAND:
            settling_time = time
        max_omega_deg = max(max_omega_deg, abs(np.degrees(omega)))

        # ── STEP 10: Log ──────────────────────────────────────
        time_log.append(time)
        theta_deg_log.append(pos_deg)
        theta_rad_log.append(theta)
        omega_log.append(omega)
        omega_deg_log.append(np.degrees(omega))
        error_deg_log.append(error_deg)
        tau_pid_log.append(tau_pid)
        tau_cmd_log.append(tau_cmd)
        tau_gravity_log.append(tau_grav)
        alpha_log.append(alpha)
        P_term_log.append(P_term)
        I_term_log.append(I_term)
        D_term_log.append(D_term)
        saturated_log.append(is_sat)

    sat_count    = sum(saturated_log)
    sat_pct      = sat_count / TIME_STEPS * 100
    final_err    = np.degrees(SETPOINT_RAD - theta)

    return {
        'label'         : label,
        'torque_cap'    : torque_cap,
        'time'          : time_log,
        'theta_deg'     : theta_deg_log,
        'omega_deg'     : omega_deg_log,
        'error_deg'     : error_deg_log,
        'tau_pid'       : tau_pid_log,
        'tau_cmd'       : tau_cmd_log,
        'tau_gravity'   : tau_gravity_log,
        'alpha'         : alpha_log,
        'P_term'        : P_term_log,
        'I_term'        : I_term_log,
        'D_term'        : D_term_log,
        'saturated'     : saturated_log,
        'settling_time' : settling_time,
        'overshoot_deg' : overshoot_deg,
        'max_omega_deg' : max_omega_deg,
        'sat_count'     : sat_count,
        'sat_pct'       : sat_pct,
        'final_err_deg' : final_err,
    }


# ============================================================
# SECTION 7 — RUN BOTH SIMULATIONS
# Run 1: With real torque cap (constrained)
# Run 2: No torque cap (unconstrained — for comparison)
# ============================================================

run_constrained   = run_simulation(TORQUE_MAX_NM, f'Constrained  (TORQUE_MAX = {TORQUE_MAX_NM} Nm)')
run_unconstrained = run_simulation(np.inf,         'Unconstrained (no torque cap)')


# ============================================================
# SECTION 8 — PRINT REPORT
# ============================================================

print("=" * 68)
print("   PHASE 2 — ACTUATOR-CONSTRAINED PID SIMULATION")
print("=" * 68)
print()
print("── SYSTEM PARAMETERS ───────────────────────────────────────")
print(f"  Joint Mass              : {JOINT_MASS} kg")
print(f"  Link Length             : {LINK_LENGTH} m")
print(f"  Moment of Inertia (I)   : {MOMENT_OF_INERTIA:.6f} kg·m²")
print(f"  Gravity at setpoint     : {GRAVITY_AT_SETPOINT:.4f} Nm  (at 90°)")
print()
print("── ACTUATOR CONSTRAINTS ────────────────────────────────────")
print(f"  Max Torque (TORQUE_MAX) : ± {TORQUE_MAX_NM} Nm")
print(f"  Max Speed  (OMEGA_MAX)  : ± {np.degrees(OMEGA_MAX_RAD_S):.1f} deg/s  ({OMEGA_MAX_RAD_S:.4f} rad/s)")
print()
print("── PID GAINS (in torque units — Nm/rad) ────────────────────")
print(f"  Kp = {Kp}    Nm/rad        (proportional)")
print(f"  Ki = {Ki}    Nm/(rad·s)    (integral — handles gravity bias)")
print(f"  Kd = {Kd}   Nm·s/rad      (derivative — reduces overshoot)")
print()
print("── SIMULATION SETTINGS ─────────────────────────────────────")
print(f"  DT         = {DT} s  (100 Hz control loop)")
print(f"  Steps      = {TIME_STEPS}      (= {TIME_STEPS * DT:.1f} s total)")
print(f"  Setpoint   = {SETPOINT_DEG}°")
print(f"  Start      = {INITIAL_DEG}°")
print(f"  Settling   = ± {SETTLING_BAND}° band")
print()

# Step-by-step log header
r = run_constrained
print("── STEP-BY-STEP LOG (constrained run, every 20 steps) ──────")
print(f"  {'Step':<6} {'Time(s)':<9} {'Pos(deg)':<11} {'Error(deg)':<13} "
      f"{'tau_cmd(Nm)':<14} {'tau_grav(Nm)':<15} {'Saturated'}")
print("  " + "-" * 72)
for i in range(0, TIME_STEPS, 20):
    sat_str = "YES <<<" if r['saturated'][i] else "no"
    print(f"  {i:<6} {r['time'][i]:<9.2f} {r['theta_deg'][i]:<11.4f} "
          f"{r['error_deg'][i]:<13.4f} {r['tau_cmd'][i]:<14.4f} "
          f"{r['tau_gravity'][i]:<15.4f} {sat_str}")

print()
print("── RESULTS COMPARISON ──────────────────────────────────────")
print(f"  {'Metric':<28} {'Constrained':>15}   {'Unconstrained':>15}")
print("  " + "-" * 62)
for key, label in [
    ('final_err_deg', 'Final error (deg)'),
    ('overshoot_deg', 'Overshoot (deg)'),
    ('settling_time', 'Settling time (s)'),
    ('max_omega_deg', 'Peak speed (deg/s)'),
    ('sat_pct',       'Time saturated (%)'),
]:
    vc = run_constrained[key]
    vu = run_unconstrained[key]
    vc_str = f"{vc:.4f}" if vc is not None else "N/A"
    vu_str = f"{vu:.4f}" if vu is not None else "N/A"
    print(f"  {label:<28} {vc_str:>15}   {vu_str:>15}")

print()
print("── SATURATION DETAIL (constrained run) ─────────────────────")
print(f"  Saturated steps : {run_constrained['sat_count']} of {TIME_STEPS} "
      f"({run_constrained['sat_pct']:.1f}% of simulation)")
sat_steps = [i for i, s in enumerate(run_constrained['saturated']) if s]
if sat_steps:
    print(f"  Saturated from  : step {sat_steps[0]} "
          f"(t={run_constrained['time'][sat_steps[0]]:.3f}s) "
          f"to step {sat_steps[-1]} "
          f"(t={run_constrained['time'][sat_steps[-1]]:.3f}s)")
    print(f"  During saturation: actuator capped at ±{TORQUE_MAX_NM} Nm")
    print(f"  → system moves slower than unconstrained case")
print()
print("── GRAVITY COMPENSATION CHECK ───────────────────────────────")
print(f"  At steady state the integral term must provide: "
      f"~{GRAVITY_AT_SETPOINT:.4f} Nm to counteract gravity at 90°")
steady_I = run_constrained['I_term'][-1]
steady_P = run_constrained['P_term'][-1]
steady_D = run_constrained['D_term'][-1]
print(f"  I_term at final step : {steady_I:.4f} Nm  "
      f"(should be ≈ {GRAVITY_AT_SETPOINT:.4f} Nm)")
print(f"  P_term at final step : {steady_P:.4f} Nm  (should be ≈ 0)")
print(f"  D_term at final step : {steady_D:.4f} Nm  (should be ≈ 0)")
print()
print("── TASK 1 vs TASK 2 — KEY DIFFERENCES ─────────────────────")
print("  TASK 1:  control = Kp*e + Ki*I + Kd*D  [dimensionless]")
print("           pos += control * DT            [no physics]")
print()
print("  TASK 2:  tau_pid = Kp*e + Ki*I + Kd*D  [Newton-metres]")
print("           tau_cmd = clip(tau_pid, ±1.5)  [actuator saturation]")
print("           alpha = (tau_cmd - tau_grav)/I [rad/s²]")
print("           omega += alpha * DT            [rad/s]")
print("           theta += omega * DT            [radians]")
print("=" * 68)


# ============================================================
# SECTION 9 — PLOTTING (5 subplots)
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Phase 2 — Actuator-Constrained PID\n'
             'Physics-Based Control: Torque → Acceleration → Position',
             fontsize=14, y=0.99)

rc  = run_constrained
ru  = run_unconstrained
t   = rc['time']

# ── Plot 1: Joint Angle vs Time (both runs) ──────────────────
ax1 = axes[0, 0]
ax1.plot(t, rc['theta_deg'], color='blue',   linewidth=2.5,
         label=f'Constrained   (TORQUE_MAX={TORQUE_MAX_NM} Nm)')
ax1.plot(t, ru['theta_deg'], color='cyan',   linewidth=1.8,
         linestyle='--', label='Unconstrained (no cap)', alpha=0.85)
ax1.axhline(y=SETPOINT_DEG,          color='red',   linestyle='--',
            linewidth=1.5, label='Setpoint (90°)')
ax1.axhline(y=SETPOINT_DEG + SETTLING_BAND, color='green', linestyle=':',
            linewidth=1.2, label=f'Settling band (±{SETTLING_BAND}°)')
ax1.axhline(y=SETPOINT_DEG - SETTLING_BAND, color='green', linestyle=':',
            linewidth=1.2)
if rc['settling_time'] is not None:
    ax1.axvline(x=rc['settling_time'], color='purple', linestyle=':',
                linewidth=1.3, alpha=0.7,
                label=f"Settled @ {rc['settling_time']:.2f}s")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Joint Angle (degrees)')
ax1.set_title('Joint Angle vs Time')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)

# ── Plot 2: Actuator Torque vs Time ─────────────────────────
ax2 = axes[0, 1]
ax2.plot(t, rc['tau_pid'],     color='gray',   linewidth=1.5,
         alpha=0.65, label='Raw PID output (tau_pid)')
ax2.plot(t, rc['tau_cmd'],     color='purple', linewidth=2.5,
         label='Clamped command (tau_cmd)')
ax2.plot(t, rc['tau_gravity'], color='orange', linewidth=2,
         linestyle='--', label='Gravity load (tau_gravity)')
ax2.axhline(y= TORQUE_MAX_NM, color='red', linestyle='--', linewidth=1.5,
            label=f'Saturation limit (±{TORQUE_MAX_NM} Nm)')
ax2.axhline(y=-TORQUE_MAX_NM, color='red', linestyle='--', linewidth=1.5)
ax2.axhline(y=0,               color='black', linestyle=':', linewidth=0.8,
            alpha=0.5)

# Shade saturation region
sat_start = next((rc['time'][i] for i, s in enumerate(rc['saturated']) if s), None)
sat_end   = next((rc['time'][i] for i, s in enumerate(rc['saturated']) if s), None)
if sat_start is not None:
    last_sat_t = max(rc['time'][i] for i, s in enumerate(rc['saturated']) if s)
    ax2.axvspan(sat_start, last_sat_t, alpha=0.12, color='red',
                label='Saturation zone')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Torque (Nm)')
ax2.set_title('Actuator Torque vs Time\n(gray=raw PID, purple=clamped, orange=gravity)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)

# ── Plot 3: Angular Velocity vs Time ────────────────────────
ax3 = axes[1, 0]
ax3.plot(t, rc['omega_deg'],  color='green',  linewidth=2.5,
         label='Angular velocity (constrained)')
ax3.plot(t, ru['omega_deg'],  color='lime',   linewidth=1.8,
         linestyle='--', alpha=0.75, label='Angular velocity (unconstrained)')
ax3.axhline(y= np.degrees(OMEGA_MAX_RAD_S), color='purple',
            linestyle=':', linewidth=1.5, label='Speed limit (±300 deg/s)')
ax3.axhline(y=-np.degrees(OMEGA_MAX_RAD_S), color='purple',
            linestyle=':', linewidth=1.5)
ax3.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angular Velocity (deg/s)')
ax3.set_title('Angular Velocity vs Time\n(shows speed limit effect)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.4)

# ── Plot 4: Error vs Time ─────────────────────────────────
ax4 = axes[1, 1]
ax4.plot(t, rc['error_deg'], color='orange', linewidth=2.5,
         label='Position error (constrained)')
ax4.plot(t, ru['error_deg'], color='gold',   linewidth=1.8,
         linestyle='--', alpha=0.75, label='Position error (unconstrained)')
ax4.axhline(y= SETTLING_BAND, color='green', linestyle=':',
            linewidth=1.3, label=f'Settling band (±{SETTLING_BAND}°)')
ax4.axhline(y=-SETTLING_BAND, color='green', linestyle=':', linewidth=1.3)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.3,
            label='Zero error')
if rc['settling_time'] is not None:
    ax4.axvline(x=rc['settling_time'], color='purple', linestyle=':',
                linewidth=1.2, alpha=0.7,
                label=f"Settled @ {rc['settling_time']:.2f}s")
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Error (degrees)')
ax4.set_title('Position Error vs Time')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.4)

# ── Plot 5: PID Component Breakdown ─────────────────────────
ax5 = axes[2, 0]
ax5.plot(t, rc['P_term'], color='blue',   linewidth=2, label='P term (Kp × error)')
ax5.plot(t, rc['I_term'], color='green',  linewidth=2, label='I term (Ki × integral)')
ax5.plot(t, rc['D_term'], color='red',    linewidth=2, label='D term (Kd × derivative)')
ax5.plot(t, rc['tau_cmd'],color='black',  linewidth=1.5, linestyle='--',
         alpha=0.7,  label='tau_cmd (clamped output)')
ax5.axhline(y= TORQUE_MAX_NM, color='gray', linestyle=':', linewidth=1.2,
            label=f'±{TORQUE_MAX_NM} Nm cap')
ax5.axhline(y=-TORQUE_MAX_NM, color='gray', linestyle=':', linewidth=1.2)
ax5.axhline(y=GRAVITY_AT_SETPOINT, color='orange', linestyle='--',
            linewidth=1.2, alpha=0.7,
            label=f'Gravity at setpoint ({GRAVITY_AT_SETPOINT:.4f} Nm)')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Torque (Nm)')
ax5.set_title('PID Component Breakdown\n(I-term converges to gravity load at steady state)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.4)

# ── Plot 6: Angular Acceleration vs Time ────────────────────
ax6 = axes[2, 1]
ax6.plot(t, rc['alpha'], color='teal', linewidth=2,
         label='Angular acceleration (alpha)')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=1.3,
            label='Zero acceleration')
ax6.fill_between(t, 0, rc['alpha'],
                 where=[a > 0 for a in rc['alpha']],
                 alpha=0.15, color='blue', label='Accelerating (positive)')
ax6.fill_between(t, 0, rc['alpha'],
                 where=[a < 0 for a in rc['alpha']],
                 alpha=0.15, color='red', label='Decelerating (negative)')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Angular Acceleration (rad/s²)')
ax6.set_title('Angular Acceleration vs Time\nalpha = (tau_cmd − tau_gravity) / I')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase2_actuator_pid_graph.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved as phase2_actuator_pid_graph.png")


# ============================================================
# SECTION 10 — PHASE 3 HANDOFF SUMMARY
# ============================================================
print()
print("=" * 68)
print("   PHASE 2 → PHASE 3 HANDOFF")
print("=" * 68)
print()
print("  Phase 2 established:")
print(f"    Torque-based PID converges to {SETPOINT_DEG}° ✓")
print(f"    Actuator saturation (±{TORQUE_MAX_NM} Nm) visible and handled ✓")
print(f"    Gravity compensation via integral term ✓")
print(f"    Physics chain: tau → alpha → omega → theta ✓")
print()
print("  Phase 3 adds:")
print("    Varying load over time (walking scenario)")
print("    Load changes mid-simulation as if robot leg is weight-bearing")
print("    PID must adapt — gravity_torque changes as simulated mass varies")
print()
print("  All Phase 1 and Phase 2 physics carry forward unchanged.")
print("=" * 68)
