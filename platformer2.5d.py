import glfw
import zengl
import numpy as np
import struct
import time
import sys
from dataclasses import dataclass
from collections import defaultdict

# verify
is_free_threaded = hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() == False
GIL_STATE = "No GIL (True Parallelism)" if is_free_threaded else "GIL Active"
print(f"Python {sys.version.split()[0]} | {GIL_STATE}")

# --- Configuration ---
WINDOW_SIZE = (1280, 720)

# --- 1. DATA: Standard Cube ---
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

# HUD Triangle
triangle_data = np.array([0.0, 0.8, 0.0, -0.1, 0.9, 0.0, 0.1, 0.9, 0.0], dtype='f4')


@dataclass
class Platform:
    x: float
    y: float
    hw: float
    hh: float

@dataclass
class Particle:
    x: float; y: float; z: float
    vx: float; vy: float; vz: float
    life: float  # Starts at 1.0, disappears at 0.0
    color: tuple


# --- 2. MATH (Verified) ---
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
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m

def get_rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    res = np.eye(4, dtype='f4')
    res[0,0] = c;  res[0,1] = s
    res[1,0] = -s; res[1,1] = c
    return res


# --- 3. PHYSICS ---
class PhysicsEngine:
    def __init__(self, platforms):
        self.player_data = np.array([0.0, 1.0, 0.0, 0.0], dtype='f4')
        self.platforms = platforms
        self.keys = {}
        self.pw, self.ph = 0.05, 0.05
        self.cell_size = 0.5
        self.grid = defaultdict(list)
        for p in self.platforms:
            min_cx = int((p.x - p.hw) // self.cell_size)
            max_cx = int((p.x + p.hw) // self.cell_size)
            min_cy = int((p.y - p.hh) // self.cell_size)
            max_cy = int((p.y + p.hh) // self.cell_size)
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    self.grid[(cx, cy)].append(p)

        self.max_particles = 2000
        self.p_pos = np.zeros((self.max_particles, 3), dtype='f4')
        self.p_vel = np.zeros((self.max_particles, 3), dtype='f4')
        self.p_life = np.zeros(self.max_particles, dtype='f4')
        self.p_idx = 0
        self.on_ground = False

    def spawn_burst(self, x, y, count, color=(0.8, 0.8, 0.8)):
        for _ in range(count):
            idx = self.p_idx % self.max_particles
            self.p_pos[idx] = [x, y, 0.0]
            # Random burst velocity
            self.p_vel[idx] = [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(1.0, 3.0),
                np.random.uniform(-1.0, 1.0)
            ]
            self.p_life[idx] = 1.0
            self.p_idx += 1

    def step(self, dt):
        px, py, vx, vy = self.player_data

        # --- TUNING CONSTANTS (Units per Second) ---
        GRAVITY = -25.0
        JUMP_FORCE = 8.5
        MOVE_ACCEL = 30.0
        FRICTION = 10.0
        MAX_FALL_SPEED = -15.0

        # 1. Apply Forces
        vy += GRAVITY * dt
        vy = max(vy, MAX_FALL_SPEED)  # Terminal velocity

        # Horizontal Friction (Damping)
        # We reduce velocity proportional to time
        vx -= vx * FRICTION * dt

        # Input
        if self.keys.get(glfw.KEY_LEFT): vx -= MOVE_ACCEL * dt
        if self.keys.get(glfw.KEY_RIGHT): vx += MOVE_ACCEL * dt

        # Jump
        if self.keys.get(glfw.KEY_UP) and self.on_ground:
            vy = JUMP_FORCE
            self.on_ground = False
            self.spawn_burst(px, py - 0.1, 20, (1.0, 1.0, 1.0))

        # 2. Integrate Position (Velocity * Time)
        # This acts as the fix for "Too Fast" movement
        new_x = px + vx * dt
        new_y = py + vy * dt

        # Reset ground (collision will set it to True if we hit floor)
        self.on_ground = False

        # Particles
        if abs(vx) > 0.5 and self.on_ground:
            self.spawn_burst(px, py - 0.1, 1, (0.4, 0.4, 0.4))

        # 3. Collision Resolution
        cx, cy = int(new_x // self.cell_size), int(new_y // self.cell_size)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for plat in self.grid.get((cx + dx, cy + dy), []):
                    rx, ry, rw, rh = plat.x, plat.y, plat.hw, plat.hh

                    diff_x = new_x - rx
                    diff_y = new_y - ry
                    overlap_x = (rw + self.pw) - abs(diff_x)
                    overlap_y = (rh + self.ph) - abs(diff_y)

                    if overlap_x > 0 and overlap_y > 0:
                        if overlap_x < overlap_y:
                            # Wall
                            new_x += overlap_x if diff_x > 0 else -overlap_x
                            vx = 0
                        else:
                            # Floor/Ceiling
                            if diff_y > 0:
                                if vy <= 0:  # Only snap if falling
                                    new_y += overlap_y
                                    vy = 0
                                    self.on_ground = True
                            else:
                                if vy > 0:  # Ceiling hit
                                    new_y -= overlap_y
                                    vy = 0

        # Respawn
        if new_y < -5.0: new_x, new_y, vx, vy = 0.0, 2.0, 0.0, 0.0

        self.player_data[:] = [new_x, new_y, vx, vy]

        # Update Particles
        active = self.p_life > 0
        if np.any(active):
            self.p_pos[active] += self.p_vel[active] * dt
            self.p_vel[active, 1] += GRAVITY * dt  # Particles obey gravity
            self.p_life[active] -= 2.0 * dt

        return new_x, new_y


# --- 4. RENDERER ---
class Renderer:
    def __init__(self, ctx, platforms):
        self.ctx = ctx
        self.platforms = platforms
        self.image = ctx.image(WINDOW_SIZE, 'rgba8unorm')
        self.depth = ctx.image(WINDOW_SIZE, 'depth24plus')

        self.image.clear_value = (0.05, 0.05, 0.1, 1.0)

        self.cam_x = 0.0
        self.cam_y = 0.0
        self.tilt_angle = 0.0

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
                    // AUTO NORMALS: This works even if winding is backwards
                    vec3 dx = dFdx(v_world_pos);
                    vec3 dy = dFdy(v_world_pos);
                    vec3 normal = normalize(cross(dx, dy));

                    vec3 sun_dir = normalize(vec3(0.5, 0.8, 1.0));
                    float diff = max(dot(normal, sun_dir), 0.0);
                    vec3 ambient = vec3(0.1, 0.1, 0.2);
                    vec3 final = v_color * (diff + 0.3) + (v_color * ambient);
                    out_color = vec4(final, 1.0);
                }
            ''',
            framebuffer=[self.image, self.depth],
            topology='triangles',

            # --- THE FIX: DISABLE CULLING ---
            # This forces the GPU to draw every face, visible or not.
            # It solves invisible tops, sides, and missing platforms.
            cull_face='none',

            depth={'func': 'less', 'write': True},
            vertex_buffers=[*zengl.bind(self.vbo, '3f', 0)],
            uniforms={
                'mvp': [0.0] * 16,
                'model': [0.0] * 16,
                'color': [1.0, 1.0, 1.0]
            },
            vertex_count=36,
        )

        self.vbo_hud = ctx.buffer(triangle_data)
        self.pipeline_2d = ctx.pipeline(
            vertex_shader='''
                #version 330 core
                layout (location = 0) in vec3 in_vert;
                void main() { gl_Position = vec4(in_vert, 1.0); }
            ''',
            fragment_shader='''
                #version 330 core
                out vec4 out_color;
                void main() { out_color = vec4(1.0, 1.0, 1.0, 1.0); }
            ''',
            framebuffer=[self.image],
            topology='triangles',
            vertex_buffers=[*zengl.bind(self.vbo_hud, '3f', 0)],
            vertex_count=3
        )

    def draw(self, player_x, player_y, dt):  # <--- Note the 'dt' argument
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        proj = get_perspective(45.0, aspect, 0.1, 100.0)

        # --- FIX: Time-Independent Camera Smoothing ---
        # This formula keeps the "0.1" feel at 60FPS, but adjusts for lag/high-fps
        # If dt is high (lag), the camera moves further to catch up.
        smooth_factor = 1.0 - 0.1 ** (dt * 60.0)

        self.cam_x += (player_x - self.cam_x) * smooth_factor
        self.cam_y += (player_y - self.cam_y) * smooth_factor

        view = get_lookat(
            np.array([self.cam_x, self.cam_y + 4.0, 6.0], dtype='f4'),
            np.array([self.cam_x, self.cam_y, 0.0], dtype='f4'),
            np.array([0.0, 1.0, 0.0], dtype='f4')
        )

        def render_obj(model, color):
            mvp = proj @ view @ model
            self.pipeline_3d.uniforms['mvp'][:] = mvp.T.tobytes()
            self.pipeline_3d.uniforms['model'][:] = model.T.tobytes()
            self.pipeline_3d.uniforms['color'][:] = struct.pack('3f', *color)
            self.pipeline_3d.render()

        for p in self.platforms:
            m = get_model_matrix(p.x, p.y, 0.0, p.hw * 2, p.hh * 2, 1.0)
            render_obj(m, (0.2, 0.8, 0.3))

        floor_m = get_model_matrix(player_x, -15.0, -10.0, 100.0, 1.0, 100.0)
        render_obj(floor_m, (0.05, 0.05, 0.1))

        # Tilt also needs time correction to be perfectly smooth
        target_tilt = -(player_x - self.cam_x) * 1.5
        self.tilt_angle += (target_tilt - self.tilt_angle) * smooth_factor

        s = np.diag([0.2, 0.2, 0.2, 1.0]).astype('f4')
        r = get_rotation_z(self.tilt_angle)
        t = np.eye(4, dtype='f4')
        t[0:3, 3] = [player_x, player_y, 0.0]

        m_p = t @ r @ s
        render_obj(m_p, (1.0, 0.5, 0.0))

        for i in range(physics.max_particles):
            life = physics.p_life[i]
            if life > 0:
                s = life * 0.08
                pos = physics.p_pos[i]
                m_part = get_model_matrix(pos[0], pos[1], pos[2], s, s, s)
                render_obj(m_part, (0.7 * life, 0.7 * life, 0.8 * life))

        self.pipeline_2d.render()
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
    glfw.swap_interval(1)

    platforms = [
        Platform(0.0, -1.0, 4.0, 0.2),
        Platform(3.0, -0.5, 0.5, 0.1),
        Platform(4.5, 0.0, 0.5, 0.1),
        Platform(6.0, 0.5, 0.5, 0.1),
    ]

    renderer = Renderer(zengl.context(), platforms)
    physics = PhysicsEngine(platforms)


    def on_key(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            physics.keys[key] = True
        elif action == glfw.RELEASE:
            physics.keys[key] = False


    glfw.set_key_callback(window, on_key)

    TARGET_FPS = 60.0
    FIXED_DT = 1.0 / TARGET_FPS
    SUB_STEPS = 8
    PHYSICS_DT = FIXED_DT / SUB_STEPS

    accumulator = 0.0
    last_time = time.perf_counter()

    # Initialize positions
    px, py = physics.player_data[0], physics.player_data[1]
    prev_px, prev_py = px, py

    while not glfw.window_should_close(window):
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        last_time = current_time

        # Prevent death spiral
        if frame_time > 0.25:
            frame_time = 0.25

        accumulator += frame_time

        while accumulator >= FIXED_DT:
            prev_px, prev_py = px, py

            for _ in range(SUB_STEPS):
                px, py = physics.step(PHYSICS_DT)

            accumulator -= FIXED_DT

        # Calculate Alpha and CLAMP it (Fixes floating point floating-over errors)
        alpha = accumulator / FIXED_DT
        alpha = max(0.0, min(1.0, alpha))

        render_x = prev_px * (1.0 - alpha) + px * alpha
        render_y = prev_py * (1.0 - alpha) + py * alpha

        # --- PASS frame_time HERE ---
        renderer.draw(render_x, render_y, frame_time)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()