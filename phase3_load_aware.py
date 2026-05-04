# ============================================================
# PHASE 3 — LOAD-AWARE CONTROL
# Varying load simulation: walking gait cycle on hip joint
# Quadruped hip joint — Mini Cheetah scale
# ============================================================
#
# WHAT THIS PHASE ADDS OVER PHASE 2:
#
#   Phase 2: Load is constant (only link gravity, 0.4905 Nm at 90deg)
#            Controller settles cleanly and stays settled.
#
#   Phase 3: Load CHANGES over time — simulating a walking gait
#
#            STANCE phase (foot on ground, robot weight on leg):
#               tau_gravity = (JOINT_MASS + LOAD_MASS) x g x L/2 x sin(theta)
#               extra load  = 0.2943 Nm at 90deg
#               total       = 0.7848 Nm at 90deg
#
#            SWING phase (leg in air, no external load):
#               tau_gravity = JOINT_MASS x g x L/2 x sin(theta)
#               total       = 0.4905 Nm at 90deg
#
#   The PID integral term must ADAPT to the changing load.
#   When load switches suddenly:
#       Integral lags behind  -> position error appears
#       Integral overcorrects when load drops -> overshoot
#
#   This is the CORE CHALLENGE of load-aware control.
#
# THREE SIMULATION RUNS:
#   Run 1 - No load  (baseline, Phase 2 equivalent)
#   Run 2 - Constant load  (0.3 kg always on joint)
#   Run 3 - Walking gait   (0.3 kg stance / 0.0 kg swing, 1.0s cycle)
#
# WALKING GAIT PARAMETERS:
#   Gait period  : 1.0 s  (1 Hz slow walk)
#   Stance phase : 60% of cycle (0.60 s) - foot on ground
#   Swing phase  : 40% of cycle (0.40 s) - foot in air
#   Gait starts  : t = 1.5 s (after system reaches setpoint)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1 - PHYSICAL PARAMETERS (from Phase 1 + 2)
# ============================================================

JOINT_MASS          = 0.5
LINK_LENGTH         = 0.20
MOMENT_ARM          = LINK_LENGTH / 2
GRAVITY             = 9.81
MOMENT_OF_INERTIA   = (1/3) * JOINT_MASS * LINK_LENGTH**2

SETPOINT_DEG        = 90.0
INITIAL_DEG         = 0.0
SETPOINT_RAD        = np.radians(SETPOINT_DEG)
INITIAL_RAD         = np.radians(INITIAL_DEG)

TORQUE_MAX_NM       = 1.5
OMEGA_MAX_RAD_S     = np.radians(300.0)

Kp = 2.0
Ki = 2.0
Kd = 0.20

DT                  = 0.01
TIME_STEPS          = 800
SETTLING_BAND_DEG   = 2.0

# ============================================================
# SECTION 2 - WALKING LOAD PARAMETERS
# ============================================================

LOAD_MASS           = 0.3      # kg  - robot body mass on hip during stance
GAIT_PERIOD         = 1.0      # s   - full gait cycle (1 Hz walk)
STANCE_FRACTION     = 0.60     # 60% stance / 40% swing
LOAD_START_STEP     = 150      # gait starts at t=1.5s (after settling)

TAU_LINK_ONLY_90    = JOINT_MASS * GRAVITY * MOMENT_ARM
TAU_WITH_LOAD_90    = (JOINT_MASS + LOAD_MASS) * GRAVITY * MOMENT_ARM
TAU_EXTRA_LOAD_90   = LOAD_MASS * GRAVITY * MOMENT_ARM


# ============================================================
# SECTION 3 - PHYSICS FUNCTIONS
# ============================================================

def gravity_torque(theta_rad, extra_mass=0.0):
    """
    tau = (m_link + m_load) x g x (L/2) x sin(theta)
    extra_mass = 0.0 during swing / no-load
    extra_mass = LOAD_MASS during stance
    """
    return (JOINT_MASS + extra_mass) * GRAVITY * MOMENT_ARM * np.sin(theta_rad)


def get_gait_load(step):
    """
    Returns (extra_mass, phase_name) for current step.
    Before LOAD_START_STEP: no load, phase = SETTLE
    After:  stance->LOAD_MASS, swing->0.0
    """
    if step < LOAD_START_STEP:
        return 0.0, 'SETTLE'
    t_since = (step - LOAD_START_STEP) * DT
    cycle_phase = (t_since % GAIT_PERIOD) / GAIT_PERIOD
    if cycle_phase < STANCE_FRACTION:
        return LOAD_MASS, 'STANCE'
    return 0.0, 'SWING'


# ============================================================
# SECTION 4 - SIMULATION RUNNER
# ============================================================

def run_simulation(load_mode, label):
    """
    load_mode: 'none' | 'constant' | 'walking'
    Returns dict of all logged data and metrics.
    """
    theta     = INITIAL_RAD
    omega     = 0.0
    integral  = 0.0
    prev_err  = SETPOINT_RAD - theta

    time_log        = []
    theta_deg_log   = []
    omega_deg_log   = []
    error_deg_log   = []
    tau_cmd_log     = []
    tau_gravity_log = []
    tau_load_log    = []
    I_term_log      = []
    P_term_log      = []
    D_term_log      = []
    load_mass_log   = []
    phase_log       = []
    saturated_log   = []

    settling_time   = None
    overshoot_deg   = 0.0
    max_err_gait    = 0.0
    gait_started    = False

    for step in range(TIME_STEPS):
        time = step * DT

        # STEP 1: Determine load
        if load_mode == 'none':
            extra_mass = 0.0
            phase_name = 'NO_LOAD'
        elif load_mode == 'constant':
            extra_mass = LOAD_MASS
            phase_name = 'CONST_LOAD'
        else:
            extra_mass, phase_name = get_gait_load(step)

        # STEP 2: PID (error in radians, output in Nm)
        error_rad   = SETPOINT_RAD - theta
        error_deg   = np.degrees(error_rad)
        integral   += error_rad * DT
        derivative  = (error_rad - prev_err) / DT

        P_term  = Kp * error_rad
        I_term  = Ki * integral
        D_term  = Kd * derivative
        tau_pid = P_term + I_term + D_term

        # STEP 3: Clamp to actuator limit
        tau_cmd = np.clip(tau_pid, -TORQUE_MAX_NM, TORQUE_MAX_NM)
        is_sat  = abs(tau_pid) > TORQUE_MAX_NM

        # STEP 4: Physics
        tau_grav  = gravity_torque(theta, extra_mass)
        tau_load  = LOAD_MASS * GRAVITY * MOMENT_ARM * np.sin(theta) \
                    if extra_mass > 0 else 0.0
        alpha     = (tau_cmd - tau_grav) / MOMENT_OF_INERTIA

        # STEP 5: Integrate
        omega    += alpha * DT
        omega     = np.clip(omega, -OMEGA_MAX_RAD_S, OMEGA_MAX_RAD_S)
        theta    += omega * DT
        prev_err  = error_rad

        # STEP 6: Metrics
        pos_deg = np.degrees(theta)
        if pos_deg > SETPOINT_DEG:
            overshoot_deg = max(overshoot_deg, pos_deg - SETPOINT_DEG)
        if settling_time is None and abs(error_deg) < SETTLING_BAND_DEG:
            settling_time = time
        if step >= LOAD_START_STEP:
            gait_started = True
        if gait_started:
            max_err_gait = max(max_err_gait, abs(error_deg))

        # STEP 7: Log
        time_log.append(time)
        theta_deg_log.append(pos_deg)
        omega_deg_log.append(np.degrees(omega))
        error_deg_log.append(error_deg)
        tau_cmd_log.append(tau_cmd)
        tau_gravity_log.append(tau_grav)
        tau_load_log.append(tau_load)
        I_term_log.append(I_term)
        P_term_log.append(P_term)
        D_term_log.append(D_term)
        load_mass_log.append(extra_mass)
        phase_log.append(phase_name)
        saturated_log.append(is_sat)

    return {
        'label'         : label,
        'load_mode'     : load_mode,
        'time'          : time_log,
        'theta_deg'     : theta_deg_log,
        'omega_deg'     : omega_deg_log,
        'error_deg'     : error_deg_log,
        'tau_cmd'       : tau_cmd_log,
        'tau_gravity'   : tau_gravity_log,
        'tau_load'      : tau_load_log,
        'I_term'        : I_term_log,
        'P_term'        : P_term_log,
        'D_term'        : D_term_log,
        'load_mass'     : load_mass_log,
        'phase'         : phase_log,
        'saturated'     : saturated_log,
        'settling_time' : settling_time,
        'overshoot_deg' : overshoot_deg,
        'max_err_gait'  : max_err_gait,
        'final_err_deg' : np.degrees(SETPOINT_RAD - theta),
        'final_pos_deg' : np.degrees(theta),
        'sat_pct'       : sum(saturated_log) / TIME_STEPS * 100,
    }


# ============================================================
# SECTION 5 - RUN SIMULATIONS
# ============================================================

run_no_load  = run_simulation('none',     'No Load (baseline)')
run_constant = run_simulation('constant', f'Constant Load ({LOAD_MASS} kg always)')
run_walking  = run_simulation('walking',  f'Walking Gait ({LOAD_MASS} kg stance / 0 swing)')


# ============================================================
# SECTION 6 - PRINT REPORT
# ============================================================

print("=" * 70)
print("   PHASE 3 - LOAD-AWARE CONTROL SIMULATION")
print("=" * 70)
print()
print("-- SYSTEM PARAMETERS (from Phase 1+2) ----------------------")
print(f"  Joint Mass              : {JOINT_MASS} kg")
print(f"  Link Length             : {LINK_LENGTH} m")
print(f"  Moment of Inertia       : {MOMENT_OF_INERTIA:.6f} kg.m2")
print(f"  Torque limit            : +/-{TORQUE_MAX_NM} Nm")
print(f"  Speed limit             : +/-{np.degrees(OMEGA_MAX_RAD_S):.0f} deg/s")
print()
print("-- WALKING LOAD MODEL --------------------------------------")
print(f"  Load mass               : {LOAD_MASS} kg")
print(f"  Gait period             : {GAIT_PERIOD} s  ({1/GAIT_PERIOD:.1f} Hz walk)")
print(f"  Stance fraction         : {STANCE_FRACTION*100:.0f}%  ({GAIT_PERIOD*STANCE_FRACTION:.2f}s per cycle)")
print(f"  Swing  fraction         : {(1-STANCE_FRACTION)*100:.0f}%  ({GAIT_PERIOD*(1-STANCE_FRACTION):.2f}s per cycle)")
print(f"  Gait starts at          : t = {LOAD_START_STEP * DT:.1f} s")
print()
print("-- GRAVITY TORQUE COMPARISON --------------------------------")
print(f"  At 90deg no load        : {TAU_LINK_ONLY_90:.4f} Nm")
print(f"  At 90deg with load      : {TAU_WITH_LOAD_90:.4f} Nm  (+{TAU_EXTRA_LOAD_90:.4f} Nm extra)")
print(f"  Actuator headroom       : {TORQUE_MAX_NM - TAU_WITH_LOAD_90:.4f} Nm above max load")
print()
print("-- STEP-BY-STEP LOG (walking run, every 50 steps) ----------")
r = run_walking
print(f"  {'Step':<6} {'Time(s)':<9} {'Pos(deg)':<11} {'Error(deg)':<13} "
      f"{'tau_cmd(Nm)':<13} {'Load(kg)':<10} {'Phase'}")
print("  " + "-" * 68)
for i in range(0, TIME_STEPS, 50):
    print(f"  {i:<6} {r['time'][i]:<9.2f} {r['theta_deg'][i]:<11.4f} "
          f"{r['error_deg'][i]:<13.4f} {r['tau_cmd'][i]:<13.4f} "
          f"{r['load_mass'][i]:<10.2f} {r['phase'][i]}")
print()
print("-- RESULTS COMPARISON --------------------------------------")
print(f"  {'Metric':<32} {'No Load':>11}   {'Const Load':>11}   {'Walking':>11}")
print("  " + "-" * 68)
for key, label in [
    ('final_pos_deg',  'Final position (deg)'),
    ('final_err_deg',  'Final error (deg)'),
    ('overshoot_deg',  'Max overshoot (deg)'),
    ('settling_time',  'Settling time (s)'),
    ('max_err_gait',   'Max err during gait (deg)'),
    ('sat_pct',        'Time saturated (%)'),
]:
    def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
    print(f"  {label:<32} {fmt(run_no_load[key]):>11}   "
          f"{fmt(run_constant[key]):>11}   {fmt(run_walking[key]):>11}")
print()
print("-- I-TERM ADAPTATION ANALYSIS ------------------------------")
avg_target = TAU_LINK_ONLY_90 + TAU_EXTRA_LOAD_90 * STANCE_FRACTION
print(f"  Target I-term (no load)      : ~{TAU_LINK_ONLY_90:.4f} Nm")
print(f"  Target I-term (const load)   : ~{TAU_WITH_LOAD_90:.4f} Nm")
print(f"  Target I-term (walking avg)  : ~{avg_target:.4f} Nm")
print(f"  Actual I-term end (no load)  :  {run_no_load['I_term'][-1]:.4f} Nm")
print(f"  Actual I-term end (const)    :  {run_constant['I_term'][-1]:.4f} Nm")
print(f"  Actual I-term end (walking)  :  {run_walking['I_term'][-1]:.4f} Nm")
print()
print("-- WHY WALKING GAIT CAUSES OSCILLATION ---------------------")
print("  1. System settles: I-term = 0.4905 Nm (gravity compensation)")
print("  2. Gait starts: stance adds 0.2943 Nm extra load at 90deg")
print("  3. I-term too small -> joint drops below setpoint")
print("  4. I-term grows -> but then swing phase hits")
print("  5. Load drops  -> I-term too large -> joint overshoots")
print("  6. Cycle repeats -> PID oscillates around setpoint")
print("  Root cause: integral lag under fast switching load")
print("=" * 70)


# ============================================================
# SECTION 7 - PLOTTING (4 subplots)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Phase 3 - Load-Aware Control\n'
             'Walking Gait: Stance/Swing Load Switching on Hip Joint',
             fontsize=14, y=0.99)

t = run_walking['time']

def shade_gait(ax, alpha=0.07):
    t_start = LOAD_START_STEP * DT
    t_end   = TIME_STEPS * DT
    t_val   = t_start
    phase   = (t_val % GAIT_PERIOD) / GAIT_PERIOD
    in_s    = phase < STANCE_FRACTION
    prev_t  = t_val
    seg_dt  = DT * 10
    while t_val < t_end:
        phase    = ((t_val - t_start) % GAIT_PERIOD) / GAIT_PERIOD
        new_in_s = phase < STANCE_FRACTION
        if new_in_s != in_s:
            c = 'red' if in_s else 'steelblue'
            ax.axvspan(prev_t, t_val, alpha=alpha, color=c, linewidth=0)
            prev_t = t_val
            in_s   = new_in_s
        t_val += seg_dt
    c = 'red' if in_s else 'steelblue'
    ax.axvspan(prev_t, t_end, alpha=alpha, color=c, linewidth=0)

# -- Plot 1: Joint Angle vs Time ------------------------------
ax1 = axes[0, 0]
ax1.plot(t, run_no_load['theta_deg'],  color='green',  linewidth=1.8,
         linestyle='--', label='No load (baseline)', alpha=0.75)
ax1.plot(t, run_constant['theta_deg'], color='orange', linewidth=1.8,
         linestyle='--', label=f'Constant load ({LOAD_MASS}kg)', alpha=0.75)
ax1.plot(t, run_walking['theta_deg'],  color='blue',   linewidth=2.5,
         label='Walking gait')
ax1.axhline(y=SETPOINT_DEG, color='red', linestyle='--',
            linewidth=1.5, label='Setpoint (90deg)')
ax1.axhline(y=SETPOINT_DEG + SETTLING_BAND_DEG, color='gray',
            linestyle=':', linewidth=1.2, label=f'+/-{SETTLING_BAND_DEG}deg band')
ax1.axhline(y=SETPOINT_DEG - SETTLING_BAND_DEG, color='gray',
            linestyle=':', linewidth=1.2)
ax1.axvline(x=LOAD_START_STEP * DT, color='black', linestyle=':',
            linewidth=1.3, alpha=0.6, label='Gait starts (t=1.5s)')
shade_gait(ax1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Joint Angle (degrees)')
ax1.set_title('Joint Angle vs Time\n(red shading=stance  blue shading=swing)')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

# -- Plot 2: Torque Components vs Time ------------------------
ax2 = axes[0, 1]
ax2.plot(t, run_walking['tau_gravity'], color='red',    linewidth=2.0,
         label='tau_gravity (link + load)')
ax2.plot(t, run_no_load['tau_gravity'], color='orange', linewidth=1.8,
         linestyle='--', alpha=0.8, label='tau_gravity (no load)')
ax2.plot(t, run_walking['tau_load'],    color='purple', linewidth=1.8,
         linestyle='-.', label='tau_load (extra stance only)')
ax2.plot(t, run_walking['tau_cmd'],     color='blue',   linewidth=2.0,
         label='tau_cmd (actuator output)')
ax2.axhline(y=TORQUE_MAX_NM, color='black', linestyle='--', linewidth=1.2,
            alpha=0.6, label=f'+/-{TORQUE_MAX_NM}Nm limit')
ax2.axhline(y=-TORQUE_MAX_NM, color='black', linestyle='--', linewidth=1.2,
            alpha=0.6)
ax2.axvline(x=LOAD_START_STEP * DT, color='black', linestyle=':',
            linewidth=1.3, alpha=0.6)
shade_gait(ax2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Torque (Nm)')
ax2.set_title('Torque Components vs Time\n(actuator tracks switching gravity load)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# -- Plot 3: Error vs Time ------------------------------------
ax3 = axes[1, 0]
ax3.plot(t, run_no_load['error_deg'],  color='green',  linewidth=1.8,
         linestyle='--', label='No load', alpha=0.75)
ax3.plot(t, run_constant['error_deg'], color='orange', linewidth=1.8,
         linestyle='--', label='Constant load', alpha=0.75)
ax3.plot(t, run_walking['error_deg'],  color='blue',   linewidth=2.5,
         label='Walking gait')
ax3.axhline(y= SETTLING_BAND_DEG, color='gray', linestyle=':',
            linewidth=1.2, label=f'+/-{SETTLING_BAND_DEG}deg band')
ax3.axhline(y=-SETTLING_BAND_DEG, color='gray', linestyle=':', linewidth=1.2)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.3, label='Zero error')
ax3.axvline(x=LOAD_START_STEP * DT, color='black', linestyle=':',
            linewidth=1.3, alpha=0.6, label='Gait starts')
shade_gait(ax3)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Error (degrees)')
ax3.set_title('Position Error vs Time\n(walking causes recurring error at each switch)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# -- Plot 4: I-term Adaptation --------------------------------
ax4 = axes[1, 1]
ax4.plot(t, run_no_load['I_term'],  color='green',  linewidth=1.8,
         linestyle='--', label='I-term: No load', alpha=0.75)
ax4.plot(t, run_constant['I_term'], color='orange', linewidth=1.8,
         linestyle='--', label='I-term: Const load', alpha=0.75)
ax4.plot(t, run_walking['I_term'],  color='blue',   linewidth=2.5,
         label='I-term: Walking gait')
avg_target = TAU_LINK_ONLY_90 + TAU_EXTRA_LOAD_90 * STANCE_FRACTION
ax4.axhline(y=TAU_LINK_ONLY_90, color='green', linestyle=':',
            linewidth=1.2, alpha=0.8,
            label=f'Target no load = {TAU_LINK_ONLY_90:.4f} Nm')
ax4.axhline(y=TAU_WITH_LOAD_90, color='red',   linestyle=':',
            linewidth=1.2, alpha=0.8,
            label=f'Target full load = {TAU_WITH_LOAD_90:.4f} Nm')
ax4.axhline(y=avg_target, color='purple', linestyle=':',
            linewidth=1.2, alpha=0.8,
            label=f'Target avg = {avg_target:.4f} Nm')
ax4.axvline(x=LOAD_START_STEP * DT, color='black', linestyle=':',
            linewidth=1.3, alpha=0.6, label='Gait starts')
shade_gait(ax4)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Integral Term (Nm)')
ax4.set_title('I-term Adaptation vs Time\n(lags load changes -> root cause of oscillation)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('phase3_load_aware_graph.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved as phase3_load_aware_graph.png")


# ============================================================
# SECTION 8 - PHASE 4 HANDOFF
# ============================================================
print()
print("=" * 70)
print("   PHASE 3 -> PHASE 4 HANDOFF")
print("=" * 70)
print()
print("  Phase 3 established:")
print("    Physics-based load switching working correctly      [OK]")
print("    I-term adapts to average load (not instantaneous)  [OK]")
print(f"    Walking gait causes +/-{run_walking['max_err_gait']:.2f}deg oscillation     [OK]")
print("    Demonstrates real challenge of varying load         [OK]")
print()
print("  Phase 4 adds:")
print("    Second joint: the KNEE")
print("    Hip + knee coupling: hip angle changes knee load")
print("    Two independent PIDs sharing the load model")
print()
print("  Constants carried forward:")
print(f"    JOINT_MASS={JOINT_MASS}kg  LINK_LENGTH={LINK_LENGTH}m")
print(f"    MOI={MOMENT_OF_INERTIA:.6f}kg.m2  TORQUE_MAX={TORQUE_MAX_NM}Nm")
print(f"    Kp={Kp}  Ki={Ki}  Kd={Kd}")
print("=" * 70)
