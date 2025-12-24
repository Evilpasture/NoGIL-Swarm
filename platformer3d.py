import glfw
import zengl
import numpy as np
import struct
import time
import sys
from dataclasses import dataclass

# verify
is_free_threaded = hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() == False
GIL_STATE = "No GIL (True Parallelism)" if is_free_threaded else "GIL Active"
print(f"Python {sys.version.split()[0]} | {GIL_STATE}")

# --- Configuration ---
WINDOW_SIZE = (1280, 720)
MOUSE_SENSITIVITY = 0.003

# --- 1. DATA ---
cube_vertices = np.array([
    # Back face
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    # Front face
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
    # Left face
    -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
    # Right face
    0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
    0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
    # Bottom face
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5,
    0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
    # Top face
    -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5,
], dtype='f4')


@dataclass
class Platform:
    x: float
    y: float
    z: float
    hw: float
    hh: float
    hd: float

@dataclass
class Particle:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    life: float  # Starts at 1.0, disappears at 0.0
    color: tuple


# --- 2. MATH ---
def get_perspective(fovy_deg, aspect, near, far):
    fovy = np.radians(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    res = np.zeros((4, 4), dtype='f4')
    res[0, 0] = f / aspect
    res[1, 1] = f
    res[2, 2] = (far + near) / (near - far)
    res[2, 3] = (2.0 * far * near) / (near - far)
    res[3, 2] = -1.0
    return res


def get_lookat(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    res = np.eye(4, dtype='f4')
    res[0, :3] = x
    res[1, :3] = y
    res[2, :3] = z
    res[0, 3] = -np.dot(x, eye)
    res[1, 3] = -np.dot(y, eye)
    res[2, 3] = -np.dot(z, eye)
    return res


def get_model_matrix(x, y, z, sx, sy, sz):
    m = np.eye(4, dtype='f4')
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


# --- 3. PHYSICS ---
class PhysicsEngine:
    def __init__(self, platforms):
        self.player_data = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype='f4')
        self.platforms = platforms
        self.keys = {}

        # Player dimensions
        self.pw, self.ph, self.pd = 0.2, 0.2, 0.2
        self.on_ground = False
        self.cam_yaw = -1.57
        self.cam_pitch = -0.3

        # Particles
        self.max_particles = 2000
        self.p_pos = np.zeros((self.max_particles, 3), dtype='f4')
        self.p_vel = np.zeros((self.max_particles, 3), dtype='f4')
        self.p_life = np.zeros(self.max_particles, dtype='f4')
        self.p_idx = 0

    def spawn_burst(self, x, y, z, count, color=(0.8, 0.8, 0.8)):
        for _ in range(count):
            idx = self.p_idx % self.max_particles
            self.p_pos[idx] = [x, y, z]

            # Random velocity in all 3 directions
            self.p_vel[idx] = [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(1.0, 3.0),
                np.random.uniform(-1.0, 1.0)
            ]
            self.p_life[idx] = 1.0
            self.p_idx += 1

    def step(self, dt):
        px, py, pz, vx, vy, vz = self.player_data

        GRAVITY = -30.0
        JUMP_FORCE = 10.0
        MOVE_ACCEL = 50.0
        FRICTION = 10.0
        MAX_FALL_SPEED = -20.0

        # Forces
        vy += GRAVITY * dt
        vy = max(vy, MAX_FALL_SPEED)
        vx -= vx * FRICTION * dt
        vz -= vz * FRICTION * dt

        # Input
        for_x, for_z = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        right_x, right_z = np.cos(self.cam_yaw + np.pi / 2), np.sin(self.cam_yaw + np.pi / 2)

        acc_x, acc_z = 0.0, 0.0
        if self.keys.get(glfw.KEY_W): acc_x += for_x; acc_z += for_z
        if self.keys.get(glfw.KEY_S): acc_x -= for_x; acc_z -= for_z
        if self.keys.get(glfw.KEY_A): acc_x -= right_x; acc_z -= right_z
        if self.keys.get(glfw.KEY_D): acc_x += right_x; acc_z += right_z

        if acc_x != 0 or acc_z != 0:
            length = np.sqrt(acc_x ** 2 + acc_z ** 2)
            vx += (acc_x / length) * MOVE_ACCEL * dt
            vz += (acc_z / length) * MOVE_ACCEL * dt

        if self.keys.get(glfw.KEY_SPACE) and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 20, (1.0, 1.0, 1.0))
            vy = JUMP_FORCE
            self.on_ground = False

        # --- Collision ---

        # X Axis
        px += vx * dt
        for plat in self.platforms:
            if self.check_overlap(px, py, pz, plat, vy=vy, strict=False):
                if px < plat.x:
                    px = plat.x - plat.hw - self.pw - 0.001
                else:
                    px = plat.x + plat.hw + self.pw + 0.001
                vx = 0

        # Z Axis
        pz += vz * dt
        for plat in self.platforms:
            if self.check_overlap(px, py, pz, plat, vy=vy, strict=False):
                if pz < plat.z:
                    pz = plat.z - plat.hd - self.pd - 0.001
                else:
                    pz = plat.z + plat.hd + self.pd + 0.001
                vz = 0

        # Y Axis (Strict Mode)
        py += vy * dt
        self.on_ground = False

        speed = np.sqrt(vx ** 2 + vz ** 2)
        if speed > 0.5 and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 1, (0.4, 0.4, 0.4))

        for plat in self.platforms:
            if self.check_overlap(px, py, pz, plat, vy=vy, strict=True):
                if vy < 0:  # Landing
                    py = plat.y + plat.hh + self.ph
                    vy = 0
                    self.on_ground = True
                elif vy > 0:  # Hitting Head (Bonk)
                    py = plat.y - plat.hh - self.ph
                    vy = 0

        # Void Respawn
        if py < -15.0:
            px, py, pz = 0.0, 2.0, 0.0
            vx, vy, vz = 0.0, 0.0, 0.0

        self.player_data[:] = [px, py, pz, vx, vy, vz]

        active = self.p_life > 0
        if np.any(active):
            self.p_pos[active] += self.p_vel[active] * dt
            self.p_vel[active, 1] += GRAVITY * dt  # Particles obey gravity
            self.p_life[active] -= 2.0 * dt

        return px, py, pz

    def check_overlap(self, x, y, z, p, vy=0.0, strict=True):
        p_min_x, p_max_x = x - self.pw, x + self.pw
        p_min_y, p_max_y = y - self.ph, y + self.ph
        p_min_z, p_max_z = z - self.pd, z + self.pd

        plat_min_x, plat_max_x = p.x - p.hw, p.x + p.hw
        plat_min_y, plat_max_y = p.y - p.hh, p.y + p.hh
        plat_min_z, plat_max_z = p.z - p.hd, p.z + p.hd

        if p_max_x <= plat_min_x or p_min_x >= plat_max_x: return False
        if p_max_y <= plat_min_y or p_min_y >= plat_max_y: return False
        if p_max_z <= plat_min_z or p_min_z >= plat_max_z: return False

        if strict or vy > 0: return True  # Walls are solid if jumping or doing Y-check

        # Step Allowance (Auto-climb small heights)
        if p_min_y >= (plat_max_y - 0.15):
            return False

        return True


# --- 4. RENDERER ---
class Renderer:
    def __init__(self, ctx, platforms):
        self.ctx = ctx
        self.platforms = platforms
        self.image = ctx.image(WINDOW_SIZE, 'rgba8unorm')
        self.depth = ctx.image(WINDOW_SIZE, 'depth24plus')
        self.image.clear_value = (0.1, 0.1, 0.15, 1.0)

        self.vbo = ctx.buffer(cube_vertices)
        self.pipeline_3d = ctx.pipeline(
            vertex_shader='''
                #version 330 core
                layout (location = 0) in vec3 in_vert;
                uniform mat4 mvp;
                uniform mat4 model;
                uniform vec3 color;
                out vec3 v_color;
                out vec3 v_world_pos;
                void main() {
                    v_color = color;
                    v_world_pos = vec3(model * vec4(in_vert, 1.0));
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core
                out vec4 out_color;
                in vec3 v_color;
                in vec3 v_world_pos;
                void main() {
                    vec3 dx = dFdx(v_world_pos);
                    vec3 dy = dFdy(v_world_pos);
                    vec3 normal = normalize(cross(dx, dy));
                    vec3 sun_dir = normalize(vec3(0.5, 0.8, 0.3));
                    float diff = max(dot(normal, sun_dir), 0.0);
                    vec3 ambient = vec3(0.2, 0.2, 0.3);
                    out_color = vec4(v_color * (diff + 0.5) + (v_color * ambient), 1.0);
                }
            ''',
            framebuffer=[self.image, self.depth],
            topology='triangles',
            cull_face='none',
            depth={'func': 'less', 'write': True},
            vertex_buffers=[*zengl.bind(self.vbo, '3f', 0)],
            uniforms={'mvp': [0.0] * 16, 'model': [0.0] * 16, 'color': [1.0, 1.0, 1.0]},
            vertex_count=36,
        )

        # Smooth camera tracking
        self.cam_x, self.cam_y, self.cam_z = 0.0, 3.0, 5.0

    def draw(self, px, py, pz, cam_yaw, cam_pitch, dt):
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        proj = get_perspective(60.0, aspect, 0.1, 100.0)

        dist = 6.0
        target_cam_x = px - np.cos(cam_yaw) * np.cos(cam_pitch) * dist
        target_cam_y = py - np.sin(cam_pitch) * dist + 1.0
        target_cam_z = pz - np.sin(cam_yaw) * np.cos(cam_pitch) * dist

        # Smooth camera catch-up
        smooth = 1.0 - 0.01 ** (dt * 6.0)
        self.cam_x += (target_cam_x - self.cam_x) * smooth
        self.cam_y += (target_cam_y - self.cam_y) * smooth
        self.cam_z += (target_cam_z - self.cam_z) * smooth

        view = get_lookat(
            np.array([self.cam_x, self.cam_y, self.cam_z], dtype='f4'),
            np.array([px, py + 0.5, pz], dtype='f4'),
            np.array([0.0, 1.0, 0.0], dtype='f4')
        )

        def render_obj(model, color):
            mvp = proj @ view @ model
            self.pipeline_3d.uniforms['mvp'][:] = mvp.T.tobytes()
            self.pipeline_3d.uniforms['model'][:] = model.T.tobytes()
            self.pipeline_3d.uniforms['color'][:] = struct.pack('3f', *color)
            self.pipeline_3d.render()

        for p in self.platforms:
            m = get_model_matrix(p.x, p.y, p.z, p.hw * 2, p.hh * 2, p.hd * 2)
            render_obj(m, (0.3, 0.7, 0.4))

        m_p = get_model_matrix(px, py, pz, 0.4, 0.4, 0.4)
        render_obj(m_p, (1.0, 0.5, 0.0))

        for i in range(physics.max_particles):
            life = physics.p_life[i]
            if life > 0:
                s = life * 0.08
                pos = physics.p_pos[i]
                m_part = get_model_matrix(pos[0], pos[1], pos[2], s, s, s)
                render_obj(m_part, (0.7 * life, 0.7 * life, 0.8 * life))

        self.image.blit()
        self.ctx.end_frame()


if __name__ == "__main__":
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WINDOW_SIZE[0], WINDOW_SIZE[1], "ZenGL 3D Platformer", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    if glfw.raw_mouse_motion_supported():
        glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)

    platforms = [
        Platform(0.0, -1.0, 0.0, 4.0, 0.2, 4.0),
        Platform(0.0, 0.5, -5.0, 1.5, 0.2, 1.5),
        Platform(-5.0, 1.5, 0.0, 1.0, 0.2, 1.0),
        Platform(5.0, 0.0, 2.0, 1.0, 0.2, 3.0),
        Platform(0.0, 2.5, -8.0, 0.5, 0.1, 0.5),
        Platform(0.0, 3.5, 5.0, 4.0, 0.2, 0.5),  # Wide wall to test climbing
    ]

    renderer = Renderer(zengl.context(), platforms)
    physics = PhysicsEngine(platforms)


    def on_key(win, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)
        if action == glfw.PRESS:
            physics.keys[key] = True
        elif action == glfw.RELEASE:
            physics.keys[key] = False


    def on_mouse(win, xpos, ypos):
        if not hasattr(on_mouse, 'last_x'):
            on_mouse.last_x, on_mouse.last_y = xpos, ypos
            return

        dx = xpos - on_mouse.last_x
        dy = ypos - on_mouse.last_y
        on_mouse.last_x, on_mouse.last_y = xpos, ypos

        physics.cam_yaw += dx * MOUSE_SENSITIVITY
        physics.cam_pitch += dy * MOUSE_SENSITIVITY
        # Clamp pitch
        physics.cam_pitch = max(-1.5, min(1.5, physics.cam_pitch))


    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_mouse)

    # LOOP CONFIG
    TARGET_FPS = 60.0
    FIXED_DT = 1.0 / TARGET_FPS
    SUB_STEPS = 4
    PHYSICS_DT = FIXED_DT / SUB_STEPS

    accumulator = 0.0
    last_time = time.perf_counter()

    px, py, pz = physics.player_data[0:3]
    prev_px, prev_py, prev_pz = px, py, pz

    while not glfw.window_should_close(window):
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        last_time = current_time
        if frame_time > 0.25: frame_time = 0.25

        accumulator += frame_time

        while accumulator >= FIXED_DT:
            prev_px, prev_py, prev_pz = px, py, pz
            for _ in range(SUB_STEPS):
                px, py, pz = physics.step(PHYSICS_DT)
            accumulator -= FIXED_DT

        alpha = accumulator / FIXED_DT
        alpha = max(0.0, min(1.0, alpha))

        rx = prev_px * (1.0 - alpha) + px * alpha
        ry = prev_py * (1.0 - alpha) + py * alpha
        rz = prev_pz * (1.0 - alpha) + pz * alpha

        # We pass camera data from physics (updated by mouse) to renderer
        renderer.draw(rx, ry, rz, physics.cam_yaw, physics.cam_pitch, frame_time)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()