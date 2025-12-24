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
# (Winding order doesn't matter anymore because we disabled culling)
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
    m[0, 0] = sx;
    m[1, 1] = sy;
    m[2, 2] = sz
    m[0, 3] = x;
    m[1, 3] = y;
    m[2, 3] = z
    return m


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

    def step(self, dt):
        px, py, vx, vy = self.player_data
        vy += -0.4 * dt
        vx *= 0.85
        if self.keys.get(glfw.KEY_LEFT): vx -= 0.005
        if self.keys.get(glfw.KEY_RIGHT): vx += 0.005
        new_x, new_y = px + vx, py + vy
        on_ground = False
        cx, cy = int(new_x // self.cell_size), int(new_y // self.cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat in self.grid.get((cx + dx, cy + dy), []):
                    rx, ry, rw, rh = plat.x, plat.y, plat.hw, plat.hh
                    if (abs(new_x - rx) < (rw + self.pw)) and (abs(new_y - ry) < (rh + self.ph)):
                        if abs(px - rx) < (rw + self.pw - 0.02):
                            if vy < 0 and py >= (ry + rh - 0.02):
                                new_y, vy, on_ground = ry + rh + self.ph, 0, True
                            elif vy > 0 and py <= (ry - rh + 0.02):
                                new_y, vy = ry - rh - self.ph, 0
                        elif abs(py - ry) < (rh + self.ph - 0.02):
                            new_x, vx = (rx - rw - self.pw if vx > 0 else rx + rw + self.pw), 0
        if self.keys.get(glfw.KEY_UP) and on_ground: vy = 0.15
        if new_y < -5.0: new_x, new_y, vx, vy = 0.0, 2.0, 0.0, 0.0
        self.player_data[:] = [new_x, new_y, vx, vy]
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

    def draw(self, player_x, player_y):
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        proj = get_perspective(45.0, aspect, 0.1, 100.0)

        self.cam_x += (player_x - self.cam_x) * 0.1
        self.cam_y += (player_y - self.cam_y) * 0.1

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

        # Draw a dark "Abyss Floor" way below
        floor_m = get_model_matrix(player_x, -15.0, -10.0, 100.0, 1.0, 100.0)
        render_obj(floor_m, (0.05, 0.05, 0.1))  # Dark Blue floor

        m_p = get_model_matrix(player_x, player_y, 0.0, 0.2, 0.2, 0.2)
        render_obj(m_p, (1.0, 0.5, 0.0))

        self.pipeline_2d.render()
        self.image.blit()
        self.ctx.end_frame()


if __name__ == "__main__":
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WINDOW_SIZE[0], WINDOW_SIZE[1], "ZenGL 3D Platformer Solved", None, None)
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

    last_time = time.perf_counter()
    while not glfw.window_should_close(window):
        now = time.perf_counter()
        px, py = physics.step(0.016)
        physics.step(0.016)
        renderer.draw(px, py)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()