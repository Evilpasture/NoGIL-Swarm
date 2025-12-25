import glfw
import zengl
import numpy as np
import struct
import time
import sys
import os
import json
from dataclasses import dataclass

# verify
is_free_threaded = hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() == False
GIL_STATE = "No GIL (True Parallelism)" if is_free_threaded else "GIL Active"
print(f"Python {sys.version.split()[0]} | {GIL_STATE}")

# --- Configuration ---
WINDOW_SIZE = (1280, 720)
MOUSE_SENSITIVITY = 0.003
SCROLL_SENSITIVITY = 1.0

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


def ray_aabb_intersect(origin, direction, box_min, box_max):
    """
    Slab Method for Ray-AABB intersection.
    Returns distance to hit, or infinity if no hit.
    """
    sign = np.sign(direction)
    sign[sign == 0] = 1.0  # Handle exact zero

    # safe_dir is never smaller than 1e-9, and always has correct sign
    safe_dir = sign * np.maximum(np.abs(direction), 1e-9)
    inv_dir = 1.0 / safe_dir

    t0 = (box_min - origin) * inv_dir
    t1 = (box_max - origin) * inv_dir

    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)

    # Largest tmin (entry point)
    t_enter = np.max(tmin)
    # Smallest tmax (exit point)
    t_exit = np.min(tmax)

    if t_exit >= t_enter and t_exit > 0:
        return t_enter
    return float('inf')


# --- 3. PHYSICS ---
class PhysicsEngine:
    def __init__(self, platforms):
        self.player_data = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype='f4')
        self.platforms = platforms
        self.keys = {}

        # Player dimensions
        self.pw, self.ph, self.pd = 0.2, 0.2, 0.2
        self.on_ground = False

        # Camera State
        self.cam_yaw = -1.57
        self.cam_pitch = -0.3
        self.target_zoom = 6.0  # Desired distance (set by scroll)
        self.current_zoom = 6.0  # Physical distance (clamped by walls)

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
        ACCEL = 50.0
        AIR_ACCEL = 10.0

        FRICTION = 10.0 if self.on_ground else 0.5
        current_accel = ACCEL if self.on_ground else AIR_ACCEL

        vy += GRAVITY * dt
        vy = max(vy, -30.0)
        damping = 1.0 / (1.0 + FRICTION * dt)
        vx *= damping
        vz *= damping

        for_x, for_z = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        right_x, right_z = np.cos(self.cam_yaw + np.pi / 2), np.sin(self.cam_yaw + np.pi / 2)
        ix, iz = 0.0, 0.0
        if self.keys.get(glfw.KEY_W): ix += for_x; iz += for_z
        if self.keys.get(glfw.KEY_S): ix -= for_x; iz -= for_z
        if self.keys.get(glfw.KEY_A): ix -= right_x; iz -= right_z
        if self.keys.get(glfw.KEY_D): ix += right_x; iz += right_z

        if ix != 0 or iz != 0:
            l = np.sqrt(ix * ix + iz * iz)
            vx += (ix / l) * current_accel * dt
            vz += (iz / l) * current_accel * dt

        if self.keys.get(glfw.KEY_SPACE) and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 20)
            vy = JUMP_FORCE
            self.on_ground = False

        # --- COLLISION LOGIC ---
        VERTICAL_ALLOWANCE = 0.25

        # 1. X-AXIS
        px += vx * dt
        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                feet_y = py - self.ph
                plat_top = p.y + p.hh
                if feet_y >= plat_top - VERTICAL_ALLOWANCE: continue
                head_y = py + self.ph
                plat_bot = p.y - p.hh
                if head_y <= plat_bot + VERTICAL_ALLOWANCE: continue

                if px < p.x:
                    px = p.x - p.hw - self.pw - 0.001
                else:
                    px = p.x + p.hw + self.pw + 0.001
                vx = 0

        # 2. Z-AXIS
        pz += vz * dt
        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                feet_y = py - self.ph
                plat_top = p.y + p.hh
                if feet_y >= plat_top - VERTICAL_ALLOWANCE: continue
                head_y = py + self.ph
                plat_bot = p.y - p.hh
                if head_y <= plat_bot + VERTICAL_ALLOWANCE: continue

                if pz < p.z:
                    pz = p.z - p.hd - self.pd - 0.001
                else:
                    pz = p.z + p.hd + self.pd + 0.001
                vz = 0

        # 3. Y-AXIS
        prev_py = py
        py += vy * dt
        self.on_ground = False

        speed = np.sqrt(vx * vx + vz * vz)
        if speed > 0.5 and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 1)

        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                if vy <= 0:
                    feet_y = py - self.ph
                    plat_top = p.y + p.hh
                    if feet_y >= plat_top - VERTICAL_ALLOWANCE:
                        py = plat_top + self.ph
                        vy = 0
                        self.on_ground = True
                elif vy > 0:
                    head_y = py + self.ph
                    plat_bot = p.y - p.hh
                    if head_y <= plat_bot + VERTICAL_ALLOWANCE:
                        py = plat_bot - self.ph
                        vy = 0

        if py < -15.0:
            px, py, pz = 0.0, 2.0, 0.0
            vx, vy, vz = 0.0, 0.0, 0.0

        self.player_data[:] = [px, py, pz, vx, vy, vz]

        active = self.p_life > 0
        if np.any(active):
            self.p_pos[active] += self.p_vel[active] * dt
            self.p_vel[active, 1] += GRAVITY * dt
            self.p_life[active] -= 2.0 * dt

        # --- CAMERA RAYCAST LOGIC (SPHERE-AABB PROXY) ---
        pivot_pos = np.array([px, py + 0.5, pz], dtype='f4')
        cam_dir_x = np.cos(self.cam_yaw) * np.cos(self.cam_pitch)
        cam_dir_y = np.sin(self.cam_pitch)
        cam_dir_z = np.sin(self.cam_yaw) * np.cos(self.cam_pitch)
        cam_dir = np.array([cam_dir_x, cam_dir_y, cam_dir_z], dtype='f4')

        # Ray direction (Backwards from pivot)
        ray_dir = -cam_dir
        closest_hit = self.target_zoom

        # FIX: Treat camera as a sphere by expanding obstacles
        CAM_RADIUS = 0.25

        for p in self.platforms:
            # Expand the platform box by the camera radius.
            # A ray hit on this expanded box == A sphere hit on the original box.
            b_min = np.array([p.x - p.hw - CAM_RADIUS, p.y - p.hh - CAM_RADIUS, p.z - p.hd - CAM_RADIUS])
            b_max = np.array([p.x + p.hw + CAM_RADIUS, p.y + p.hh + CAM_RADIUS, p.z + p.hd + CAM_RADIUS])

            dist = ray_aabb_intersect(pivot_pos, ray_dir, b_min, b_max)

            if dist < closest_hit:
                closest_hit = dist

        # Don't let zoom get too close (inside player's head)
        self.current_zoom = max(0.4, closest_hit)

        return px, py, pz

    def check_overlap(self, x, y, z, p):
        return not (
                x + self.pw <= p.x - p.hw or
                x - self.pw >= p.x + p.hw or
                y + self.ph <= p.y - p.hh or
                y - self.ph >= p.y + p.hh or
                z + self.pd <= p.z - p.hd or
                z - self.pd >= p.z + p.hd
        )


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

    def draw(self, px, py, pz, cam_yaw, cam_pitch, zoom_dist, dt):
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        proj = get_perspective(60.0, aspect, 0.1, 100.0)

        # Calculate Camera Position based on Zoom
        # Note: We rely on the 'zoom_dist' passed from physics engine
        # which has already checked for wall collisions.
        target_cam_x = px - np.cos(cam_yaw) * np.cos(cam_pitch) * zoom_dist
        target_cam_y = py - np.sin(cam_pitch) * zoom_dist + 1.0  # +1.0 offset (pivot height)
        target_cam_z = pz - np.sin(cam_yaw) * np.cos(cam_pitch) * zoom_dist

        smooth = 1.0 - 0.01 ** (dt * 10.0)  # Faster camera follow
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
    glfw.swap_interval(1)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    if glfw.raw_mouse_motion_supported():
        glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
    platforms = []
    if os.path.exists("level.json"):
        with open("level.json", "r") as f:
            data = json.load(f)
            for d in data: platforms.append(Platform(**d))
        print("Loaded level.json")
    else:
        # Default level
        platforms = [Platform(0.0, -1.0, 0.0, 4.0, 0.2, 4.0)]

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


    # New Scroll Callback
    def on_scroll(win, xoff, yoff):
        physics.target_zoom -= yoff * SCROLL_SENSITIVITY
        # Clamp zoom between 1.0 (close) and 15.0 (far)
        physics.target_zoom = max(1.0, min(15.0, physics.target_zoom))


    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_mouse)
    glfw.set_scroll_callback(window, on_scroll)

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

        # Pass current_zoom (calculated in physics step) to renderer
        renderer.draw(rx, ry, rz, physics.cam_yaw, physics.cam_pitch, physics.current_zoom, frame_time)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()