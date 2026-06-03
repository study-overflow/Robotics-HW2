"""
Heart-shaped trajectory demo + MP4 recording.
Uses admittance control to keep human force <50N.
Records MuJoCo rendering with force overlay.

Author: Zhang Zhang (2023010916)
"""

import mujoco
import numpy as np
import cv2
import os
import sys
import glfw
import subprocess
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import ComputedTorqueController


def heart_curve(t, scale=0.9, freq=0.15):
    """Heart curve in XZ plane. Returns (pos, vel, acc)."""
    w = 2 * np.pi * freq
    phase = w * t
    sin_t = np.sin(phase)
    cos_t = np.cos(phase)

    # Position
    x_raw = 16.0 * sin_t**3
    z_raw = 13.0 * cos_t - 5.0 * np.cos(2*phase) \
            - 2.0 * np.cos(3*phase) - np.cos(4*phase)
    s = scale / 16.0
    pos = np.array([x_raw * s, z_raw * s])

    # Velocity
    x_dot = 48.0 * sin_t**2 * cos_t * w * s
    z_dot = (-13.0*sin_t + 10.0*np.sin(2*phase)
             + 6.0*np.sin(3*phase) + 4.0*np.sin(4*phase)) * w * s
    vel = np.array([x_dot, z_dot])

    # Acceleration
    x_ddot = 48.0 * w**2 * (2*sin_t*cos_t**2 - sin_t**3) * s
    z_ddot = (-13.0*cos_t + 20.0*np.cos(2*phase)
              + 18.0*np.cos(3*phase) + 16.0*np.cos(4*phase)) * w**2 * s
    acc = np.array([x_ddot, z_ddot])

    return pos, vel, acc


def main():
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.xml')
    total_time = 16.0      # slower: ~2.4 hearts in 16s
    fps = 30
    width, height = 1280, 720
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'figures', 'heart_trajectory.mp4')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_steps = int(total_time / dt)

    # Computed torque + feedforward for precise heart tracking
    ct = ComputedTorqueController(mass=100.0, gravity=9.81,
                                   kp=(900.0, 900.0), kv=(60.0, 60.0))

    # For force display: compute equivalent human force via inverse admittance
    M_d = np.array([3.0, 3.0])
    D_d = np.array([35.0, 35.0])

    # Camera: Y-axis top-down view (looking along Y at XZ plane)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.0, 0.0, 0.0])
    cam.distance = 5.0
    cam.azimuth = 90      # camera on +Y axis
    cam.elevation = 0     # looking horizontally along Y at XZ plane

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    # Hidden GLFW window for offscreen GL context
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(width, height, "offscreen", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW offscreen window failed")
    glfw.make_context_current(window)

    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Output
    frame_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'figures', 'heart_frames')
    os.makedirs(frame_dir, exist_ok=True)

    traj_history = []
    force_history = []

    print(f"Recording heart demo (Y-axis top-down, admittance control, {total_time}s)...")
    frame_count = 0
    frame_interval = int(1.0 / (fps * dt))

    for step in range(n_steps):
        t = step * dt

        # Heart trajectory (pos, vel, acc)
        q_d, v_d, a_d = heart_curve(t, scale=0.9, freq=0.15)

        # Equivalent human force (for display only — inverse admittance)
        F_h = M_d * a_d + D_d * v_d
        F_h_mag = np.linalg.norm(F_h)

        # Computed torque with acceleration feedforward
        q = np.array([data.qpos[0], data.qpos[1]])
        qdot = np.array([data.qvel[0], data.qvel[1]])
        error = q_d - q
        derror = v_d - qdot
        tau = ct.mass * (ct.Kp * error + ct.Kv * derror + a_d)
        tau[1] += ct.mass * ct.gravity
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]

        mujoco.mj_step(model, data)

        traj_history.append(q.copy())
        force_history.append(F_h.copy())

        # Render at video fps
        if step % frame_interval == 0:
            mujoco.mjv_updateScene(model, data, opt, None, cam,
                                    mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)

            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewport, context)
            rgb = np.ascontiguousarray(np.flipud(rgb))

            # Draw trajectory overlay
            if len(traj_history) > 1:
                px_per_m = height / 5.0
                for i in range(1, len(traj_history), 3):  # every 3rd point for speed
                    prev = traj_history[i-1]
                    curr = traj_history[i]
                    u_prev = int(width/2 + prev[0] * px_per_m)
                    v_prev = int(height/2 - prev[1] * px_per_m)
                    u_curr = int(width/2 + curr[0] * px_per_m)
                    v_curr = int(height/2 - curr[1] * px_per_m)
                    alpha = i / len(traj_history)
                    color = (int(60+195*alpha), int(100+100*alpha), int(255-100*alpha))
                    cv2.line(rgb, (u_prev, v_prev), (u_curr, v_curr), color, 2)

            # Text overlay
            cv2.putText(rgb, f"Heart Trajectory  |  t = {t:.1f}s  |  freq = 0.15Hz",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(rgb, f"Equiv. Human Force: Fx={F_h[0]:+6.1f}N  Fz={F_h[1]:+6.1f}N  |F|={F_h_mag:5.1f}N  [<50N]",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(rgb, f"Position: x={q[0]:+.3f}m  z={q[1]:+.3f}m",
                        (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(rgb, "Computed Torque + Acceleration FF  |  Force shown: equiv. human force via inv. admittance  |  Y-axis top-down view",
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            fname = os.path.join(frame_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frame_count += 1

    glfw.destroy_window(window)
    glfw.terminate()

    # Encode with ffmpeg (H.264, universally playable)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'fast', '-crf', '23', video_path
    ], capture_output=True)
    shutil.rmtree(frame_dir)
    print(f"  -> Saved: {video_path}  ({frame_count} frames, {total_time}s)")

    # Metrics
    traj = np.array(traj_history)
    forces = np.array(force_history)
    max_force = np.max(np.linalg.norm(forces, axis=1))
    print(f"  Max human force: {max_force:.1f}N")

    # Plot result
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    t_plot = np.linspace(0, total_time, 500)
    heart_desired = np.array([heart_curve(ti, scale=0.9, freq=0.15)[0] for ti in t_plot])

    # Trajectory
    ax = axes[0, 0]
    ax.plot(heart_desired[:, 0], heart_desired[:, 1], 'b--', lw=1.5, label='Desired heart')
    ax.plot(traj[:, 0], traj[:, 1], 'r-', lw=1.5, label='Actual')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=120, marker='o', zorder=5,
               edgecolors='darkgreen', lw=2, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=120, marker='s', zorder=5,
               edgecolors='darkblue', lw=2, label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
    ax.set_title('Heart Trajectory (computed torque + acc FF, equiv. F_h < 50N)')
    ax.axis('equal'); ax.legend(); ax.grid(True, alpha=0.3)

    # Error
    ax = axes[0, 1]
    t_actual = np.linspace(0, total_time, len(traj))
    desired_at_t = np.array([heart_curve(ti, scale=0.9, freq=0.15)[0] for ti in t_actual])
    err_x = (traj[:, 0] - desired_at_t[:, 0]) * 1000
    err_z = (traj[:, 1] - desired_at_t[:, 1]) * 1000
    ax.plot(t_actual, err_x, 'r-', lw=1.0, label='Error X')
    ax.plot(t_actual, err_z, 'b-', lw=1.0, label='Error Z')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Tracking Error [mm]')
    ax.set_title('Heart Tracking Error')
    ax.legend(); ax.grid(True, alpha=0.3)
    rms_x = np.sqrt(np.mean((err_x/1000)**2))
    rms_z = np.sqrt(np.mean((err_z/1000)**2))
    ax.text(0.02, 0.98, f'RMS: X={rms_x*1000:.1f}mm, Z={rms_z*1000:.1f}mm',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Force
    ax = axes[1, 0]
    ax.plot(t_actual, forces[:, 0], 'r-', lw=1.2, label='$F_{h,x}$')
    ax.plot(t_actual, forces[:, 1], 'b-', lw=1.2, label='$F_{h,z}$')
    ax.axhline(y=50, color='gray', ls=':', alpha=0.5)
    ax.axhline(y=-50, color='gray', ls=':', alpha=0.5)
    ax.fill_between([0, total_time], -50, 50, color='green', alpha=0.03)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Human Force [N]')
    ax.set_title(f'Human Force (max |F|={max_force:.1f}N < 50N)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Position over time
    ax = axes[1, 1]
    ax.plot(t_actual, traj[:, 0], 'r-', lw=1.2, label='x(t)')
    ax.plot(t_actual, traj[:, 1], 'b-', lw=1.2, label='z(t)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position [m]')
    ax.set_title('Position vs Time')
    ax.legend(); ax.grid(True, alpha=0.3)

    png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'figures', 'heart_trajectory.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {png_path}")
    print(f"  RMS tracking error: X={rms_x*1000:.2f}mm, Z={rms_z*1000:.2f}mm")


if __name__ == "__main__":
    main()
