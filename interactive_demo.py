"""
Interactive MuJoCo Demo — Mouse-Driven Admittance Control
==========================================================
Drag the 100kg cube with your mouse. The admittance controller amplifies
your small force so the heavy cube moves easily.

Controls:
  Left mouse drag   → apply force to cube
  Right mouse drag  → rotate view
  Scroll            → zoom
  R                 → reset cube to origin
  1/2/3             → auto-demo (horizontal / vertical / diagonal)
  SPACE             → toggle manual / auto
  ESC               → quit
"""

import mujoco
import mujoco.glfw
import numpy as np
import glfw
import os


class AdmittanceController:
    def __init__(self, dt, mass_desired=(3.0, 3.0), damping_desired=(35.0, 35.0),
                 joint_limits=((-1.5, 1.5), (-1.5, 1.5))):
        self.dt = dt
        self.M_d = np.array(mass_desired)
        self.D_d = np.array(damping_desired)
        self.limits = joint_limits
        self.x_d = np.zeros(2)
        self.v_d = np.zeros(2)

    def update(self, F_h):
        a_d = (F_h - self.D_d * self.v_d) / self.M_d
        self.v_d += a_d * self.dt
        self.x_d += self.v_d * self.dt
        margin = 0.05
        self.x_d[0] = np.clip(self.x_d[0], self.limits[0][0] + margin, self.limits[0][1] - margin)
        self.x_d[1] = np.clip(self.x_d[1], self.limits[1][0] + margin, self.limits[1][1] - margin)
        return self.x_d.copy(), self.v_d.copy()

    def reset(self, x0=(0.0, 0.0)):
        self.x_d = np.array(x0, dtype=float)
        self.v_d = np.zeros(2)


class ComputedTorqueController:
    def __init__(self, mass=100.0, gravity=9.81, kp=(900.0, 900.0), kv=(60.0, 60.0)):
        self.mass = mass
        self.gravity = gravity
        self.Kp = np.array(kp)
        self.Kv = np.array(kv)

    def compute_control(self, q, qdot, q_d, qdot_d):
        error = q_d - q
        derror = qdot_d - qdot
        qddot_d = self.Kp * error + self.Kv * derror
        tau = self.mass * qddot_d
        tau[1] += self.mass * self.gravity
        return tau


class InteractiveDemo:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # FIX 1: More responsive admittance (lighter feel, less damping)
        self.admittance = AdmittanceController(self.dt,
                                                mass_desired=(3.0, 3.0),
                                                damping_desired=(35.0, 35.0))
        self.ct = ComputedTorqueController()

        # Mouse state
        self.mouse_down = False
        self.mouse_force = np.zeros(2)
        self.mouse_raw = np.zeros(2)
        self.last_cursor = None

        # Auto-demo state
        self.auto_mode = 0
        self.auto_t = 0.0

        # FIX 2: Camera — top-down view along Y axis (俯视图)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat = np.array([0.0, 0.0, 0.0])
        self.cam.distance = 5.0
        self.cam.azimuth = 90       # look from +X direction
        self.cam.elevation = 0   # true top-down (俯视图)

        self.opt = mujoco.MjvOption()

        # Rendering
        self.window = None
        self.scene = None
        self.context = None
        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0

    # ---- Callbacks ----
    def keyboard_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.admittance.reset()
            mujoco.mj_forward(self.model, self.data)
            self.mouse_raw = np.zeros(2)
            self.mouse_force = np.zeros(2)
            print("[Reset]")
        elif key == glfw.KEY_1:
            self.auto_mode = 1; self.auto_t = 0.0
            self.mouse_raw[:] = 0; self.mouse_force[:] = 0
            print("[Auto] Horizontal")
        elif key == glfw.KEY_2:
            self.auto_mode = 2; self.auto_t = 0.0
            self.mouse_raw[:] = 0; self.mouse_force[:] = 0
            print("[Auto] Vertical")
        elif key == glfw.KEY_3:
            self.auto_mode = 3; self.auto_t = 0.0
            self.mouse_raw[:] = 0; self.mouse_force[:] = 0
            print("[Auto] Diagonal")
        elif key == glfw.KEY_SPACE:
            if self.auto_mode > 0:
                self.auto_mode = 0
                self.mouse_raw[:] = 0; self.mouse_force[:] = 0
                print("[Manual] Mouse drag")
            else:
                self.auto_mode = 1; self.auto_t = 0.0
                print("[Auto]")

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_down = True
                xpos, ypos = glfw.get_cursor_pos(window)
                self.last_cursor = (xpos, ypos)
                self.auto_mode = 0
            else:
                self.mouse_down = False
                self.last_cursor = None

    def cursor_pos_callback(self, window, xpos, ypos):
        if self.mouse_down and self.last_cursor is not None:
            dx = xpos - self.last_cursor[0]
            dy = ypos - self.last_cursor[1]

            # FIX 3: More sensitive mouse-to-force conversion
            scale = 0.8       # N/pixel  (was 0.15, now much more responsive)
            self.mouse_raw[0] += dx * scale    # horizontal mouse → X
            self.mouse_raw[1] -= dy * scale    # up mouse → +Z (lift up)

            # FIX 4: Much gentler decay so force persists between frames
            self.mouse_raw *= 0.95   # was 0.85, now decays slower

            # Clamp to 50N
            self.mouse_raw[0] = np.clip(self.mouse_raw[0], -50, 50)
            self.mouse_raw[1] = np.clip(self.mouse_raw[1], -50, 50)

        self.last_cursor = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        self.cam.distance *= (1.0 - yoffset * 0.1)
        self.cam.distance = np.clip(self.cam.distance, 1.0, 20.0)

    def _auto_force(self, t):
        period = 8.0
        tp = t % period
        amp = 40.0
        if tp < 0.15 * period:
            return amp * 0.5 * (1.0 - np.cos(np.pi * tp / (0.15 * period)))
        elif tp < 0.45 * period:
            return amp
        elif tp < 0.50 * period:
            tau = (tp - 0.45 * period) / (0.05 * period)
            return amp * 0.5 * (1.0 + np.cos(np.pi * tau))
        elif tp < 0.65 * period:
            tau = (tp - 0.50 * period) / (0.15 * period)
            return -amp * 0.5 * (1.0 - np.cos(np.pi * tau))
        elif tp < 0.95 * period:
            return -amp
        elif tp < period:
            tau = (tp - 0.95 * period) / (0.05 * period)
            return -amp * 0.5 * (1.0 + np.cos(np.pi * tau))
        return 0.0

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        width, height = 1400, 900
        self.window = glfw.create_window(width, height,
                                          "Admittance Control — 100kg Cube (俯视图)",
                                          None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Render context (after GL context is current)
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Callbacks
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        mujoco.mjv_defaultCamera(self.cam)
        # Override for top-down view (use FREE camera, not FIXED)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.elevation = 0
        self.cam.azimuth = 90
        self.cam.distance = 5.0
        self.cam.lookat = np.array([0.0, 0.0, 0.0])
        mujoco.mjv_defaultOption(self.opt)

        sim_time = 0.0
        print("\n" + "=" * 60)
        print("  Admittance Control Demo — Top-Down View (俯视图)")
        print("  " + "-" * 40)
        print("  LEFT DRAG   →  apply force to cube")
        print("  RIGHT DRAG  →  rotate view")
        print("  SCROLL      →  zoom")
        print("  1/2/3       →  auto demo (H / V / diagonal)")
        print("  R           →  reset")
        print("  ESC         →  quit")
        print("=" * 60)

        while not glfw.window_should_close(self.window):
            sim_time += self.dt
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))

            # Force input
            if self.auto_mode > 0:
                self.auto_t += self.dt
                f = self._auto_force(self.auto_t)
                if self.auto_mode == 1:
                    F_h = np.array([f, 0.0])
                elif self.auto_mode == 2:
                    F_h = np.array([0.0, f * 45.0 / 40.0])
                else:
                    F_h = np.array([f * 0.8, f * 0.8])
            else:
                # Smooth mouse force
                alpha = 0.3
                self.mouse_force = alpha * self.mouse_raw + (1 - alpha) * self.mouse_force
                F_h = self.mouse_force.copy()

            # Control
            x_d, v_d = self.admittance.update(F_h)
            q = np.array([self.data.qpos[0], self.data.qpos[1]])
            qdot = np.array([self.data.qvel[0], self.data.qvel[1]])
            tau = self.ct.compute_control(q, qdot, x_d, v_d)
            self.data.ctrl[0] = tau[0]
            self.data.ctrl[1] = tau[1]

            mujoco.mj_step(self.model, self.data)

            # Render
            mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert,
                                    self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)

            # Overlay
            mode_names = {0: "MANUAL (drag cube)", 1: "AUTO: Horizontal",
                          2: "AUTO: Vertical", 3: "AUTO: Diagonal"}
            lines = [
                f"{mode_names.get(self.auto_mode, '')}  |  Time: {sim_time:.1f}s",
                f"Human force:  Fx={F_h[0]:+5.1f}N  Fz={F_h[1]:+5.1f}N  |F|={np.linalg.norm(F_h):5.1f}N  [max 50N]",
                f"Position:     x={q[0]:+6.3f}m  z={q[1]:+6.3f}m",
                f"Actuator:    tx={self.data.actuator_force[0]:+7.0f}N  tz={self.data.actuator_force[1]:+7.0f}N",
                f"Keys: 1/2/3=auto  R=reset  SPACE=toggle  ESC=quit",
            ]
            for i, line in enumerate(lines):
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                    viewport, line, "", self.context)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
        print("Demo ended.")


if __name__ == "__main__":
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.xml')
    demo = InteractiveDemo(xml_path)
    demo.run()
