"""
Cartesian Robot Admittance Control Simulation
==============================================
Problem: Manipulate a 100kg cube in a vertical plane using a Cartesian robot.
The human applies <50N force, and the robot assists via admittance control.

Control Architecture:
  Outer loop (Admittance):  M_d * v_d' + D_d * v_d = F_h
  Inner loop (Computed Torque): tau = M * (Kp*(q_d-q) + Kv*(qdot_d-qdot)) + G(q)

Author: Zhang Zhang (2023010916)
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import time

rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})


# ============================================================
# 1. Admittance Controller (Outer Loop)
# ============================================================
class AdmittanceController:
    """
    Converts human force F_h into desired motion via virtual dynamics.

        M_d * dv_d/dt + D_d * v_d = F_h

    where M_d, D_d are the desired apparent mass and damping.
    Small M_d makes the 100kg object feel light.
    D_d provides velocity-proportional resistance for stability.
    """

    def __init__(self, dt, mass_desired=(5.0, 5.0), damping_desired=(80.0, 80.0),
                 joint_limits=((-1.5, 1.5), (-1.5, 1.5))):
        self.dt = dt
        self.M_d = np.array(mass_desired)
        self.D_d = np.array(damping_desired)
        self.joint_limits = joint_limits

        self.x_d = np.zeros(2)
        self.v_d = np.zeros(2)

    def update(self, F_h):
        """
        Args:
            F_h: [Fx, Fz] human force in world frame [N]

        Returns:
            x_d: desired position [x, z] in world frame
        """
        # Admittance: a_d = (F_h - D_d * v_d) / M_d
        a_d = (F_h - self.D_d * self.v_d) / self.M_d

        # Euler integration
        self.v_d += a_d * self.dt
        self.x_d += self.v_d * self.dt

        # Clamp desired position to joint limits with margin
        margin = 0.05
        self.x_d[0] = np.clip(self.x_d[0],
                              self.joint_limits[0][0] + margin,
                              self.joint_limits[0][1] - margin)
        self.x_d[1] = np.clip(self.x_d[1],
                              self.joint_limits[1][0] + margin,
                              self.joint_limits[1][1] - margin)

        return self.x_d.copy(), self.v_d.copy()

    def reset(self, x0=(0.0, 0.0)):
        self.x_d = np.array(x0, dtype=float)
        self.v_d = np.zeros(2)


# ============================================================
# 2. Human Force Generator (Test Input)
# ============================================================
class HumanForceGenerator:
    """Generates simulated human force profiles (< 50N)."""

    @staticmethod
    def horizontal_sweep(t, total_time=8.0):
        """
        Horizontal: push right, then left.
        Smooth force profile using sinusoidal ramps.
        """
        Fx = _smooth_bidirectional(t, total_time, amplitude=40.0,
                                   ramp_frac=0.15, hold_frac=0.30, pause_frac=0.05)
        return np.array([Fx, 0.0])

    @staticmethod
    def vertical_sweep(t, total_time=8.0):
        """
        Vertical: lift up, then push down.
        """
        Fz = _smooth_bidirectional(t, total_time, amplitude=45.0,
                                   ramp_frac=0.15, hold_frac=0.30, pause_frac=0.05)
        return np.array([0.0, Fz])

    @staticmethod
    def diagonal_sweep(t, total_time=8.0):
        """
        Composite: up-right, then down-left.
        """
        F_mag = _smooth_bidirectional(t, total_time, amplitude=35.0,
                                       ramp_frac=0.15, hold_frac=0.30, pause_frac=0.05)
        # Split force equally between x and z to demonstrate composite motion
        return np.array([F_mag * 0.8, F_mag * 0.8])


def _smooth_bidirectional(t, total_time, amplitude=40.0,
                          ramp_frac=0.15, hold_frac=0.30, pause_frac=0.05):
    """
    Generate a smooth bidirectional force profile.

    Phase 0: ramp up positive  (0 -> ramp_frac)
    Phase 1: hold positive      (ramp_frac -> ramp_frac+hold_frac)
    Phase 2: ramp down to zero  (ramp_frac+hold_frac -> ramp_frac+hold_frac+pause_frac)
    Phase 3: pause at zero      (brief pause)
    Phase 4: ramp up negative   (symmetrical)
    Phase 5: hold negative
    Phase 6: ramp down to zero
    Phase 7: pause at zero      (rest of time)

    Uses smooth cos-based ramps rather than linear.
    """
    T1 = total_time * ramp_frac          # ramp up positive
    T2 = total_time * (ramp_frac + hold_frac)  # end of hold positive
    T3 = total_time * (ramp_frac + hold_frac + pause_frac)  # ramp down to zero
    T_mid = total_time * 0.5             # midpoint
    T4 = T_mid + T1                      # ramp up negative
    T5 = T_mid + T2                      # end of hold negative
    T6 = T_mid + T3                      # ramp down to zero

    t_norm = np.clip(t, 0, total_time)

    if t_norm < T1:
        # Ramp up positive (smooth cos ramp)
        return amplitude * 0.5 * (1.0 - np.cos(np.pi * t_norm / T1))
    elif t_norm < T2:
        # Hold positive
        return amplitude
    elif t_norm < T3:
        # Ramp down to zero
        tau = (t_norm - T2) / (T3 - T2)
        return amplitude * 0.5 * (1.0 + np.cos(np.pi * tau))
    elif t_norm < T_mid:
        # Pause at zero
        return 0.0
    elif t_norm < T4:
        # Ramp up negative
        return -amplitude * 0.5 * (1.0 - np.cos(np.pi * (t_norm - T_mid) / T1))
    elif t_norm < T5:
        # Hold negative
        return -amplitude
    elif t_norm < T6:
        # Ramp down to zero
        tau = (t_norm - T5) / (T6 - T5)
        return -amplitude * 0.5 * (1.0 + np.cos(np.pi * tau))
    else:
        return 0.0


# ============================================================
# 3. Computed-Torque Inner-Loop Controller
# ============================================================
class ComputedTorqueController:
    """
    Inner-loop position controller using computed torque with
    gravity compensation.

        tau = M * (Kp * e + Kv * e_dot) + G

    where:
        M = diag(100, 100) kg (actual mass)
        G = [0, m*g] (gravity vector for Z axis)
        e = q_d - q (position error)
    """

    def __init__(self, mass=100.0, gravity=9.81,
                 kp=(900.0, 900.0), kv=(60.0, 60.0)):
        self.mass = mass
        self.gravity = gravity
        self.Kp = np.array(kp)
        self.Kv = np.array(kv)

    def compute_control(self, q_actual, qdot_actual, q_desired, qdot_desired):
        """
        Compute joint torques for the robot.

        Args:
            q_actual:     actual joint positions [x, z]
            qdot_actual:  actual joint velocities [xdot, zdot]
            q_desired:    desired joint positions [x_d, z_d]
            qdot_desired: desired joint velocities [xdot_d, zdot_d]

        Returns:
            tau: joint torques [tau_x, tau_z]
        """
        error = q_desired - q_actual
        derror = qdot_desired - qdot_actual

        # PD acceleration
        qddot_desired = self.Kp * error + self.Kv * derror

        # Computed torque
        tau = self.mass * qddot_desired

        # Gravity compensation (only for Z axis)
        tau[1] += self.mass * self.gravity

        return tau


# ============================================================
# 4. Simulation Runner
# ============================================================
def run_simulation(xml_path, force_generator, total_time=8.0, case_name="test"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    n_steps = int(total_time / dt)

    # Initial pose
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)

    # Controllers
    admittance = AdmittanceController(dt,
                                       mass_desired=(5.0, 5.0),
                                       damping_desired=(80.0, 80.0),
                                       joint_limits=((-1.5, 1.5), (-1.5, 1.5)))
    admittance.reset(x0=(data.qpos[0], data.qpos[1]))

    ct_controller = ComputedTorqueController(
        mass=100.0, gravity=9.81,
        kp=(900.0, 900.0), kv=(60.0, 60.0)
    )

    # Data logging
    log = {
        't': np.zeros(n_steps),
        'x_actual': np.zeros((n_steps, 2)),
        'v_actual': np.zeros((n_steps, 2)),
        'x_desired': np.zeros((n_steps, 2)),
        'v_desired': np.zeros((n_steps, 2)),
        'F_human': np.zeros((n_steps, 2)),
        'actuator_force': np.zeros((n_steps, 2)),
        'tracking_error': np.zeros((n_steps, 2)),
    }

    for step in range(n_steps):
        t = step * dt

        # 1. Get human force
        F_h = force_generator(t, total_time)

        # 2. Admittance outer loop: F_h -> desired motion
        x_d, v_d = admittance.update(F_h)

        # 3. Computed-torque inner loop
        q = np.array([data.qpos[0], data.qpos[1]])
        qdot = np.array([data.qvel[0], data.qvel[1]])
        tau = ct_controller.compute_control(q, qdot, x_d, v_d)

        # 4. Apply control
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]

        # 5. Step simulation
        mujoco.mj_step(model, data)

        # 6. Log
        log['t'][step] = t
        log['x_actual'][step] = q
        log['v_actual'][step] = qdot
        log['x_desired'][step] = x_d
        log['v_desired'][step] = v_d
        log['F_human'][step] = F_h
        log['actuator_force'][step] = data.actuator_force
        log['tracking_error'][step] = x_d - q

    return log


# ============================================================
# 5. Plotting
# ============================================================
def plot_results(log, case_name, save_dir="figures"):
    import matplotlib.ticker as ticker
    os.makedirs(save_dir, exist_ok=True)
    t = log['t']

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # --- Position tracking: X ---
    ax = axes[0, 0]
    ax.plot(t, log['x_desired'][:, 0], 'b--', linewidth=2.0, label='Desired $x_d$')
    ax.plot(t, log['x_actual'][:, 0], 'r-', linewidth=1.5, label='Actual $x$')
    ax.set_ylabel('X Position [m]')
    ax.set_title('Horizontal Position Tracking')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    # --- Position tracking: Z ---
    ax = axes[0, 1]
    ax.plot(t, log['x_desired'][:, 1], 'b--', linewidth=2.0, label='Desired $z_d$')
    ax.plot(t, log['x_actual'][:, 1], 'r-', linewidth=1.5, label='Actual $z$')
    ax.set_ylabel('Z Position [m]')
    ax.set_title('Vertical Position Tracking')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    # Add a small inset to show tracking quality if error is small relative to range
    z_range = np.max(log['x_actual'][:, 1]) - np.min(log['x_actual'][:, 1])
    z_err_max = np.max(np.abs(log['tracking_error'][:, 1]))
    if z_err_max < 0.05 * z_range and z_range > 0.5:
        inset = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
        inset.plot(t, log['x_desired'][:, 1], 'b--', linewidth=1.0)
        inset.plot(t, log['x_actual'][:, 1], 'r-', linewidth=1.0)
        t0 = int(0.15 * len(t)); t1 = int(0.35 * len(t))
        inset.set_xlim(t[t0], t[t1])
        z_mid = 0.5 * (np.max(log['x_desired'][:, 1][t0:t1]) + np.min(log['x_desired'][:, 1][t0:t1]))
        inset.set_ylim(z_mid - 5*z_err_max, z_mid + 5*z_err_max)
        inset.set_title('Zoom: tracking detail', fontsize=8)
        inset.tick_params(labelsize=7)
        ax.indicate_inset_zoom(inset, edgecolor='gray')

    # --- Tracking error ---
    ax = axes[1, 0]
    ax.plot(t, log['tracking_error'][:, 0] * 1000, 'r-', linewidth=1.0, label='Error X')
    ax.plot(t, log['tracking_error'][:, 1] * 1000, 'b-', linewidth=1.0, label='Error Z')
    ax.set_ylabel('Tracking Error [mm]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Position Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Human Force ---
    ax = axes[1, 1]
    ax.plot(t, log['F_human'][:, 0], 'r-', linewidth=1.5, label='$F_{h,x}$')
    ax.plot(t, log['F_human'][:, 1], 'b-', linewidth=1.5, label='$F_{h,z}$')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=-50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.fill_between([0, t[-1]], -50, 50, color='green', alpha=0.05)
    ax.set_ylabel('Force [N]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Human Applied Force (|F| < 50 N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Actuator Forces ---
    ax = axes[2, 0]
    ax.plot(t, log['actuator_force'][:, 0], 'r-', linewidth=1.5, label=r'$\tau_x$')
    ax.plot(t, log['actuator_force'][:, 1], 'b-', linewidth=1.5, label=r'$\tau_z$')
    ax.axhline(y=981, color='gray', linestyle=':', alpha=0.5,
               label='Gravity compensation (981 N)')
    ax.set_ylabel('Actuator Force [N]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Robot Actuator Forces')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Workspace trajectory ---
    ax = axes[2, 1]
    x_traj = log['x_actual'][:, 0]
    z_traj = log['x_actual'][:, 1]
    # Draw the full trajectory
    ax.plot(x_traj, z_traj, 'gray', linewidth=1.2, alpha=0.5, zorder=1)
    # Add direction arrows along the path (every 25% of the way)
    for frac in [0.15, 0.40, 0.65, 0.90]:
        idx = int(frac * len(t))
        idx_next = min(idx + 80, len(t) - 1)
        ax.annotate('', xy=(x_traj[idx_next], z_traj[idx_next]),
                    xytext=(x_traj[idx], z_traj[idx]),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, alpha=0.7))
    # Start point
    ax.scatter(x_traj[0], z_traj[0], c='green', s=160, marker='o',
               zorder=5, edgecolors='darkgreen', linewidth=2,
               label='Start / End (return)')
    # End point
    ax.scatter(x_traj[-1], z_traj[-1], c='lime', s=100, marker='o',
               zorder=5, edgecolors='darkgreen', linewidth=1.5)
    # Find the farthest point from start (= midpoint of a back-forth path)
    dist = np.sqrt((x_traj - x_traj[0])**2 + (z_traj - z_traj[0])**2)
    farthest_idx = np.argmax(dist)
    ax.scatter(x_traj[farthest_idx], z_traj[farthest_idx],
               c='blue', s=140, marker='D', zorder=5, edgecolors='darkblue', linewidth=2,
               label=f'Farthest point (t={t[farthest_idx]:.1f}s)')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Z Position [m]')
    ax.set_title('End-Effector Trajectory in X-Z Plane\n(bidirectional, returns to origin)')
    ax.axis('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Admittance Control Simulation: {case_name}',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    fname = f'{save_dir}/{case_name.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {fname}")
    return fig


def compute_metrics(log):
    """Compute quantitative performance metrics."""
    t = log['t']

    err_x = log['tracking_error'][:, 0]
    err_z = log['tracking_error'][:, 1]

    # RMS tracking error
    rms_x = np.sqrt(np.mean(err_x**2))
    rms_z = np.sqrt(np.mean(err_z**2))
    max_err_x = np.max(np.abs(err_x))
    max_err_z = np.max(np.abs(err_z))

    # Human force statistics
    max_fh_x = np.max(np.abs(log['F_human'][:, 0]))
    max_fh_z = np.max(np.abs(log['F_human'][:, 1]))
    rms_fh_x = np.sqrt(np.mean(log['F_human'][:, 0]**2))
    rms_fh_z = np.sqrt(np.mean(log['F_human'][:, 1]**2))

    # Actuator force
    max_act_x = np.max(np.abs(log['actuator_force'][:, 0]))
    max_act_z = np.max(np.abs(log['actuator_force'][:, 1]))

    # Displacement
    disp_x = np.max(log['x_actual'][:, 0]) - np.min(log['x_actual'][:, 0])
    disp_z = np.max(log['x_actual'][:, 1]) - np.min(log['x_actual'][:, 1])

    # Average velocity during motion
    vx_mean = np.mean(np.abs(log['v_actual'][:, 0]))
    vz_mean = np.mean(np.abs(log['v_actual'][:, 1]))

    return {
        'rms_x': rms_x, 'rms_z': rms_z,
        'max_err_x': max_err_x, 'max_err_z': max_err_z,
        'max_fh_x': max_fh_x, 'max_fh_z': max_fh_z,
        'rms_fh_x': rms_fh_x, 'rms_fh_z': rms_fh_z,
        'max_act_x': max_act_x, 'max_act_z': max_act_z,
        'disp_x': disp_x, 'disp_z': disp_z,
        'vx_mean': vx_mean, 'vz_mean': vz_mean,
    }


# ============================================================
# 6. Main
# ============================================================
def main():
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.xml')

    experiments = [
        ("Horizontal Motion", HumanForceGenerator.horizontal_sweep),
        ("Vertical Motion", HumanForceGenerator.vertical_sweep),
        ("Composite Diagonal Motion", HumanForceGenerator.diagonal_sweep),
    ]

    all_metrics = {}

    print("=" * 70)
    print("  Cartesian Robot Admittance Control — MuJoCo Simulation")
    print("  Object: 100 kg cube  |  Vertical (X-Z) plane  |  g = 9.81 m/s²")
    print("  Human force limit: |F_h| < 50 N")
    print("=" * 70)

    for name, force_gen in experiments:
        print(f"\n{'─' * 70}")
        print(f"  Experiment: {name}")
        print(f"{'─' * 70}")

        t0 = time.time()
        log = run_simulation(xml_path, force_gen, total_time=8.0, case_name=name)
        elapsed = time.time() - t0
        print(f"  Wall time: {elapsed:.3f}s  |  Steps: {len(log['t'])}")

        metrics = compute_metrics(log)
        all_metrics[name] = metrics

        print(f"\n  ┌─ Tracking Performance ──────────────────────────┐")
        print(f"  │  RMS error:   X = {metrics['rms_x']*1000:6.2f} mm,  Z = {metrics['rms_z']*1000:6.2f} mm   │")
        print(f"  │  Max error:   X = {metrics['max_err_x']*1000:6.2f} mm,  Z = {metrics['max_err_z']*1000:6.2f} mm   │")
        print(f"  ├─ Human Force ───────────────────────────────────┤")
        print(f"  │  Max |F|:     X = {metrics['max_fh_x']:6.2f} N,   Z = {metrics['max_fh_z']:6.2f} N    │")
        print(f"  │  RMS |F|:     X = {metrics['rms_fh_x']:6.2f} N,   Z = {metrics['rms_fh_z']:6.2f} N    │")
        print(f"  ├─ Actuator Force ────────────────────────────────┤")
        print(f"  │  Max |τ|:     X = {metrics['max_act_x']:7.1f} N,  Z = {metrics['max_act_z']:7.1f} N   │")
        print(f"  ├─ Motion ────────────────────────────────────────┤")
        print(f"  │  Displacement: X = {metrics['disp_x']:6.2f} m,   Z = {metrics['disp_z']:6.2f} m     │")
        print(f"  │  Mean |v|:    X = {metrics['vx_mean']:6.3f} m/s, Z = {metrics['vz_mean']:6.3f} m/s  │")
        print(f"  └─────────────────────────────────────────────────┘")

        plot_results(log, name)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  Summary — All Experiments")
    print(f"{'=' * 70}")
    header = f"  {'Experiment':<28} {'RMS X':>7} {'RMS Z':>7} {'Max Err X':>8} {'Max Err Z':>8} {'|Fh|max':>7} {'|τ|max':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in all_metrics.items():
        max_fh = max(m['max_fh_x'], m['max_fh_z'])
        max_act = max(m['max_act_x'], m['max_act_z'])
        print(f"  {name:<28} {m['rms_x']*1e3:>6.1f}mm {m['rms_z']*1e3:>6.1f}mm "
              f"{m['max_err_x']*1e3:>7.1f}mm {m['max_err_z']*1e3:>7.1f}mm "
              f"{max_fh:>6.1f}N {max_act:>7.1f}N")

    print(f"\n  All figures saved to ./figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
