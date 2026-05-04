# ============================================================
# PHASE 1 — PHYSICAL SYSTEM MAPPING
# Converting angle control to real physics: torque and load
# Robotic joint actuator — quadruped hip joint model
# ============================================================
#
# WHAT THIS PHASE DOES:
#   Task 1 moved a joint in "degrees" with no physical meaning.
#   Phase 1 grounds everything in real units:
#       angle (degrees / radians)
#       → gravity torque (Newton-metres)
#       → required actuator torque (Newton-metres)
#       → angular acceleration (rad/s²)
#
#   This mapping becomes the physics engine that ALL future
#   phases (2–7) plug into instead of abstract numbers.
#
# JOINT BEING MODELLED:
#   Hip joint of a small quadruped robot (Mini Cheetah scale)
#   Controls the upper leg (thigh segment)
#   Rotates from 0° (hanging vertical) to 90° (horizontal)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1 — PHYSICAL PARAMETERS
# All values have real units and physical meaning
# ============================================================

JOINT_MASS            = 0.5               # kg     — mass of the thigh link
LINK_LENGTH           = 0.20              # m      — length of thigh segment
MOMENT_ARM            = LINK_LENGTH / 2   # m      — COM at midpoint (uniform rod)
GRAVITY               = 9.81             # m/s²   — gravitational acceleration

# Moment of inertia: I = (1/3) × m × L²
# Rotational inertia of a uniform rod rotating about one end
MOMENT_OF_INERTIA     = (1/3) * JOINT_MASS * LINK_LENGTH**2   # kg·m²

# Setpoint and initial position (consistent with Task 1)
SETPOINT_DEG          = 90.0             # degrees — target position
INITIAL_DEG           = 0.0             # degrees — starting position
SETPOINT_RAD          = np.radians(SETPOINT_DEG)
INITIAL_RAD           = np.radians(INITIAL_DEG)

# Sweep of angles for the full mapping (0° to 180°)
ANGLES_DEG            = np.linspace(0, 180, 361)
ANGLES_RAD            = np.radians(ANGLES_DEG)


# ============================================================
# SECTION 2 — PHYSICS FUNCTIONS
# ============================================================

def gravity_torque(angle_deg):
    """
    Torque produced by gravity acting on the joint link.

    Formula:
        tau_gravity = m × g × (L/2) × sin(theta)

    Angle convention:
        0°   → link hanging straight down (vertical)
               gravity acts along the link axis
               perpendicular moment arm = 0  →  tau = 0 Nm
        90°  → link horizontal
               gravity acts perpendicular to the link
               moment arm = L/2 (maximum)  →  tau = maximum
        180° → link pointing straight up (unstable upright)
               gravity again along axis  →  tau = 0 Nm
    """
    angle_rad = np.radians(angle_deg)
    return JOINT_MASS * GRAVITY * MOMENT_ARM * np.sin(angle_rad)


def required_hold_torque(angle_deg):
    """
    Minimum actuator torque needed to HOLD the joint at a given angle.
    Equal to gravity torque (static equilibrium, no acceleration, no friction).
    """
    return gravity_torque(angle_deg)


def angular_acceleration(actuator_torque_nm, angle_deg, friction_torque=0.0):
    """
    Angular acceleration produced at a given joint angle.

    Equation of motion:
        I × alpha = tau_actuator − tau_gravity − tau_friction
        alpha = (tau_actuator − tau_gravity − tau_friction) / I

    Positive alpha → joint accelerates toward higher angles
    Negative alpha → joint decelerates or moves toward lower angles
    """
    tau_gravity = gravity_torque(angle_deg)
    tau_net     = actuator_torque_nm - tau_gravity - friction_torque
    alpha       = tau_net / MOMENT_OF_INERTIA
    return alpha


def torque_to_hold_and_accelerate(angle_deg, desired_alpha_rad_s2):
    """
    Total actuator torque needed to both hold position AND accelerate.
        tau_total = tau_gravity + I × alpha_desired
    """
    return gravity_torque(angle_deg) + MOMENT_OF_INERTIA * desired_alpha_rad_s2


# ============================================================
# SECTION 3 — COMPUTE MAPPING VALUES
# ============================================================

torque_profile      = gravity_torque(ANGLES_DEG)
tau_at_initial      = gravity_torque(INITIAL_DEG)
tau_at_setpoint     = gravity_torque(SETPOINT_DEG)
tau_max             = np.max(torque_profile)
angle_at_max_torque = ANGLES_DEG[np.argmax(torque_profile)]

# Acceleration achievable across a range of actuator torques
tau_range       = np.linspace(0, 2.0, 200)
alpha_at_0deg   = [angular_acceleration(t, 0.0)  for t in tau_range]
alpha_at_45deg  = [angular_acceleration(t, 45.0) for t in tau_range]
alpha_at_90deg  = [angular_acceleration(t, 90.0) for t in tau_range]

# Torque needed to reach setpoint in 2 seconds from rest
# Using kinematic formula: theta = 0.5 * alpha * t^2 → alpha = 2*theta / t^2
target_alpha_rad_s2 = 2 * SETPOINT_RAD / (2.0 ** 2)
tau_needed_full     = torque_to_hold_and_accelerate(0.0, target_alpha_rad_s2)


# ============================================================
# SECTION 4 — PRINT REPORT
# ============================================================

print("=" * 65)
print("   PHASE 1 — PHYSICAL SYSTEM MAPPING")
print("=" * 65)
print()
print("── PHYSICAL PARAMETERS ─────────────────────────────────────")
print(f"  Joint Mass              : {JOINT_MASS}     kg")
print(f"  Link Length             : {LINK_LENGTH}    m")
print(f"  Moment Arm (L/2)        : {MOMENT_ARM}    m")
print(f"  Gravitational Accel     : {GRAVITY}   m/s²")
print(f"  Moment of Inertia (I)   : {MOMENT_OF_INERTIA:.6f} kg·m²")
print(f"  Formula: I = (1/3)×m×L² = (1/3)×{JOINT_MASS}×{LINK_LENGTH}²")
print()
print("── ANGLE CONVENTION ────────────────────────────────────────")
print("     0°  =  link hanging vertically down  (initial position)")
print("    90°  =  link horizontal               (target setpoint)")
print("   180°  =  link pointing straight up     (unstable)")
print()
print("── ANGLE → GRAVITY TORQUE MAPPING ─────────────────────────")
print(f"  {'Angle (deg)':<14} {'sin(theta)':<14} {'tau_gravity (Nm)':<20} {'Hold torque (Nm)'}")
print("  " + "-" * 62)
for angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
    s   = np.sin(np.radians(angle))
    tau = gravity_torque(angle)
    print(f"  {angle:<14} {s:<14.4f} {tau:<20.4f} {tau:.4f}")
print()
print("── KEY POSITIONS ───────────────────────────────────────────")
print(f"  At initial   (  0°): tau_gravity = {tau_at_initial:.4f} Nm  (no load)")
print(f"  At setpoint  ( 90°): tau_gravity = {tau_at_setpoint:.4f} Nm  <- controller must exceed this")
print(f"  Maximum torque at  : {angle_at_max_torque:.1f}° ->  {tau_max:.4f} Nm")
print()
print("── EQUATION OF MOTION ──────────────────────────────────────")
print("  I × alpha = tau_actuator − tau_gravity − tau_friction")
print()
print(f"  At 0°  (initial):    {MOMENT_OF_INERTIA:.6f} × alpha = tau_act − {tau_at_initial:.4f} − tau_fric")
print(f"  At 90° (setpoint):   {MOMENT_OF_INERTIA:.6f} × alpha = tau_act − {tau_at_setpoint:.4f} − tau_fric")
print()
print("── MINIMUM ACTUATOR REQUIREMENTS ──────────────────────────")
print(f"  To HOLD at 90°         : actuator must provide >= {tau_at_setpoint:.4f} Nm")
print(f"  To reach 90° in ~2s    : actuator must provide >= {tau_needed_full:.4f} Nm")
print(f"  Desired alpha          : {np.degrees(target_alpha_rad_s2):.2f} deg/s² = {target_alpha_rad_s2:.4f} rad/s²")
print()
print("── TASK 1 vs TASK 2 COMPARISON ────────────────────────────")
print("  TASK 1:  current_pos += control × DT           [abstract units]")
print("  TASK 2:  alpha = (tau_act − tau_grav) / I")
print("           omega += alpha × DT                   [rad/s]")
print("           theta += omega × DT                   [radians → degrees]")
print("=" * 65)


# ============================================================
# SECTION 5 — PLOTTING (4 subplots)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Phase 1 — Physical System Mapping\n'
             'Robotic Hip Joint: Angle → Torque → Load',
             fontsize=14, y=0.99)

# ── Plot 1: Gravity Torque vs Joint Angle ────────────────────
ax1 = axes[0, 0]
ax1.plot(ANGLES_DEG, torque_profile, color='blue', linewidth=2.5,
         label='tau_gravity = m·g·(L/2)·sin(θ)')
ax1.axvline(x=INITIAL_DEG,  color='green', linestyle='--', linewidth=1.5,
            label='Initial (0°)')
ax1.axvline(x=SETPOINT_DEG, color='red',   linestyle='--', linewidth=1.5,
            label='Setpoint (90°)')
ax1.axhline(y=tau_at_setpoint, color='orange', linestyle=':', linewidth=1.5,
            alpha=0.9, label=f'At setpoint = {tau_at_setpoint:.4f} Nm')
ax1.scatter([90], [tau_at_setpoint], color='red', zorder=5, s=90)
ax1.set_xlabel('Joint Angle (degrees)')
ax1.set_ylabel('Gravity Torque (Nm)')
ax1.set_title('Gravity Torque vs Joint Angle')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)
ax1.set_xlim(0, 180)
ax1.set_ylim(-0.02, 0.58)
ax1.annotate(f'{tau_at_setpoint:.4f} Nm', xy=(90, tau_at_setpoint),
             xytext=(105, tau_at_setpoint + 0.07),
             fontsize=8, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

# ── Plot 2: Link Geometry at Key Angles ──────────────────────
ax2 = axes[0, 1]
ax2.set_xlim(-0.07, 0.32)
ax2.set_ylim(-0.30, 0.08)
ax2.set_aspect('equal')
ax2.set_facecolor('#f7f7f7')

geo_colors   = ['#27ae60', '#2980b9', '#e67e22', '#e74c3c']
angles_shown = [0, 30, 60, 90]

for i, angle_d in enumerate(angles_shown):
    angle_r = np.radians(angle_d)
    x_tip   = LINK_LENGTH * np.sin(angle_r)
    y_tip   = -LINK_LENGTH * np.cos(angle_r)
    x_com   = MOMENT_ARM * np.sin(angle_r)
    y_com   = -MOMENT_ARM * np.cos(angle_r)

    # Draw link
    ax2.plot([0, x_tip], [0, y_tip], color=geo_colors[i],
             linewidth=3.5, alpha=0.85, label=f'{angle_d}°')
    # COM marker
    ax2.plot(x_com, y_com, 'o', color=geo_colors[i], markersize=8, zorder=5)
    # Gravity arrow at COM (scale by torque for visual clarity)
    tau = gravity_torque(angle_d)
    arrow_len = max(tau * 0.30, 0.0)
    if arrow_len > 0.005:
        ax2.annotate('', xy=(x_com, y_com - arrow_len),
                     xytext=(x_com, y_com),
                     arrowprops=dict(arrowstyle='->', color=geo_colors[i],
                                     lw=2, mutation_scale=14))

ax2.plot(0, 0, 'ks', markersize=12, zorder=10, label='Joint')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('X position (m)')
ax2.set_ylabel('Y position (m)')
ax2.set_title('Link Geometry at Key Angles\n(arrows = gravity force at COM)')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.text(0.02, -0.26, '0° = hanging vertical\n(start position)',
         fontsize=7, color=geo_colors[0])

# ── Plot 3: Min Hold Torque across Angles ────────────────────
ax3 = axes[1, 0]
ax3.fill_between(ANGLES_DEG, 0, torque_profile, alpha=0.20,
                 color='red', label='Hold torque region')
ax3.plot(ANGLES_DEG, torque_profile, color='red',
         linewidth=2.5, label='Min hold torque curve')
ax3.axvline(x=SETPOINT_DEG, color='blue', linestyle='--', linewidth=1.5,
            label=f'Setpoint 90°')
ax3.axhline(y=tau_at_setpoint, color='orange', linestyle=':',
            linewidth=1.5, alpha=0.85)
ax3.scatter([90], [tau_at_setpoint], color='blue', zorder=5, s=90)
ax3.annotate(f'Min actuator torque\nat 90° = {tau_at_setpoint:.4f} Nm',
             xy=(90, tau_at_setpoint),
             xytext=(100, tau_at_setpoint + 0.10),
             fontsize=8, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax3.set_xlabel('Joint Angle (degrees)')
ax3.set_ylabel('Required Torque (Nm)')
ax3.set_title('Min Actuator Torque to Hold Position\n'
              '(static equilibrium — no friction, no acceleration)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.4)
ax3.set_xlim(0, 180)
ax3.set_ylim(-0.02, 0.60)

# ── Plot 4: Angular Acceleration vs Actuator Torque ──────────
ax4 = axes[1, 1]
ax4.plot(tau_range, alpha_at_0deg,  color='green',  linewidth=2,
         label=f'At  0°  (tau_grav = {gravity_torque(0):.4f} Nm)')
ax4.plot(tau_range, alpha_at_45deg, color='orange', linewidth=2,
         label=f'At 45°  (tau_grav = {gravity_torque(45):.4f} Nm)')
ax4.plot(tau_range, alpha_at_90deg, color='red',    linewidth=2,
         label=f'At 90°  (tau_grav = {gravity_torque(90):.4f} Nm)')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6,
            label='Zero acceleration (hover)')
ax4.axvline(x=tau_at_setpoint, color='red', linestyle=':', linewidth=1.2,
            alpha=0.6)

# Mark hover points (where alpha = 0 for each angle)
for angle_h, color_h in [(0, 'green'), (45, 'orange'), (90, 'red')]:
    tau_hover = gravity_torque(angle_h)
    ax4.scatter([tau_hover], [0], color=color_h, zorder=6, s=80)

ax4.set_xlabel('Actuator Torque (Nm)')
ax4.set_ylabel('Angular Acceleration (rad/s²)')
ax4.set_title('Angular Acceleration vs Actuator Torque\n'
              'alpha = (tau_act − tau_grav) / I')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.4)
ax4.set_xlim(0, 2.0)

eq_text = (f'I = {MOMENT_OF_INERTIA:.6f} kg·m²\n'
           f'Dots = zero-accel (hover) points')
ax4.text(1.0, min(alpha_at_90deg) + 20, eq_text, fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.75))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('phase1_graph.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved as phase1_graph.png")


# ============================================================
# SECTION 6 — PHASE 2 HANDOFF SUMMARY
# ============================================================
print()
print("=" * 65)
print("   PHASE 1 → PHASE 2 HANDOFF")
print("=" * 65)
print()
print("  Defined and verified:")
print(f"    JOINT_MASS          = {JOINT_MASS} kg")
print(f"    LINK_LENGTH         = {LINK_LENGTH} m")
print(f"    MOMENT_ARM          = {MOMENT_ARM} m")
print(f"    GRAVITY             = {GRAVITY} m/s²")
print(f"    MOMENT_OF_INERTIA   = {MOMENT_OF_INERTIA:.6f} kg·m²")
print()
print("  Physics functions ready:")
print("    gravity_torque(angle_deg)             -> Nm")
print("    angular_acceleration(tau, angle_deg)  -> rad/s²")
print()
print("  Phase 2 will replace Task 1's:")
print("    current_pos += control × DT           [no units]")
print("  with:")
print("    alpha = (tau_act − tau_gravity) / I   [rad/s²]")
print("    omega += alpha × DT                   [rad/s]")
print("    theta += omega × DT                   [rad → degrees]")
print("=" * 65)
