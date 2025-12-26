import glfw
import zengl
import numpy as np
import time
import sys
from pathlib import Path
import json
from dataclasses import dataclass
from platformer3d.platformer3d_editor import Editor

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
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
    -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
    -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5,
], dtype='f4')


@dataclass
class Platform:
    x: float
    y: float
    z: float
    hw: float
    hh: float
    hd: float

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z, 'hw': self.hw, 'hh': self.hh, 'hd': self.hd}


# --- 2. ZERO-ALLOC MATH ---
def get_perspective(fovy_deg, aspect, near, far, out):
    out.fill(0)
    fovy = np.radians(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    # Note indices are swapped [col, row] logic relative to standard C memory
    out[0, 0] = f / aspect
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[3, 2] = (2.0 * far * near) / (near - far)
    out[2, 3] = -1.0


def get_lookat(eye, target, up, out):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    out.fill(0)
    out[0, 0] = 1
    out[1, 1] = 1
    out[2, 2] = 1
    out[3, 3] = 1

    # Rotation (Transposed)
    out[0, 0] = x[0]
    out[0, 1] = y[0]
    out[0, 2] = z[0]
    out[1, 0] = x[1]
    out[1, 1] = y[1]
    out[1, 2] = z[1]
    out[2, 0] = x[2]
    out[2, 1] = y[2]
    out[2, 2] = z[2]

    # Translation (Stored in last column / bottom row of memory)
    out[3, 0] = -np.dot(x, eye)
    out[3, 1] = -np.dot(y, eye)
    out[3, 2] = -np.dot(z, eye)


def get_model_matrix(out, x, y, z, sx, sy, sz):
    out.fill(0)
    out[0, 0] = sx
    out[1, 1] = sy
    out[2, 2] = sz
    out[3, 3] = 1.0
    # Translation (Stored in last column)
    out[3, 0] = x
    out[3, 1] = y
    out[3, 2] = z


def ray_aabb_intersect(origin, direction, box_min, box_max):
    sign = np.sign(direction)
    sign[sign == 0] = 1.0
    safe_dir = sign * np.maximum(np.abs(direction), 1e-9)
    inv_dir = 1.0 / safe_dir
    t0 = (box_min - origin) * inv_dir
    t1 = (box_max - origin) * inv_dir
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    t_enter = np.max(tmin)
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
        self.pw, self.ph, self.pd = 0.2, 0.2, 0.2
        self.on_ground = False

        # Camera State (Controlled by Physics in PLAY mode)
        self.cam_yaw = -1.57
        self.cam_pitch = -0.3
        self.target_zoom = 6.0
        self.current_zoom = 6.0

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
            self.p_vel[idx] = [np.random.uniform(-1.0, 1.0), np.random.uniform(1.0, 3.0), np.random.uniform(-1.0, 1.0)]
            self.p_life[idx] = 1.0
            self.p_idx = (self.p_idx + 1) % self.max_particles

    def step(self, dt):
        px, py, pz, vx, vy, vz = self.player_data
        GRAVITY = -30.0
        JUMP_FORCE = 10.0
        if self.on_ground:
            max_speed, accel, friction = 6.0, 50.0, 15.0
        else:
            max_speed, accel, friction = 6.0, 20.0, 0.5

        vy += GRAVITY * dt
        vy = max(vy, -30.0)
        damping = 1.0 / (1.0 + friction * dt)
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
            wish_dir_x, wish_dir_z = ix / l, iz / l
            current_speed_in_wish_dir = vx * wish_dir_x + vz * wish_dir_z
            add_speed = max_speed - current_speed_in_wish_dir
            if add_speed > 0:
                accel_speed = min(accel * dt, add_speed)
                vx += wish_dir_x * accel_speed
                vz += wish_dir_z * accel_speed

        if self.keys.get(glfw.KEY_SPACE) and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 20)
            vy = JUMP_FORCE
            self.on_ground = False

        VERTICAL_ALLOWANCE = 0.25
        px += vx * dt
        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                if py - self.ph >= p.y + p.hh - VERTICAL_ALLOWANCE: continue
                if py + self.ph <= p.y - p.hh + VERTICAL_ALLOWANCE: continue
                if px < p.x:
                    px = p.x - p.hw - self.pw - 0.001
                else:
                    px = p.x + p.hw + self.pw + 0.001
                vx = 0

        pz += vz * dt
        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                if py - self.ph >= p.y + p.hh - VERTICAL_ALLOWANCE: continue
                if py + self.ph <= p.y - p.hh + VERTICAL_ALLOWANCE: continue
                if pz < p.z:
                    pz = p.z - p.hd - self.pd - 0.001
                else:
                    pz = p.z + p.hd + self.pd + 0.001
                vz = 0

        py += vy * dt
        self.on_ground = False
        speed = np.sqrt(vx * vx + vz * vz)
        if speed > 0.5 and self.on_ground: self.spawn_burst(px, py - 0.1, pz, 1)

        for p in self.platforms:
            if self.check_overlap(px, py, pz, p):
                if vy <= 0 and py - self.ph >= p.y + p.hh - VERTICAL_ALLOWANCE:
                    py = p.y + p.hh + self.ph
                    vy = 0
                    self.on_ground = True
                elif vy > 0 and py + self.ph <= p.y - p.hh + VERTICAL_ALLOWANCE:
                    py = p.y - p.hh - self.ph
                    vy = 0

        if py < -15.0: px, py, pz, vx, vy, vz = 0.0, 2.0, 0.0, 0.0, 0.0, 0.0

        self.player_data[:] = [px, py, pz, vx, vy, vz]
        active = self.p_life > 0
        if np.any(active):
            self.p_pos[active] += self.p_vel[active] * dt
            self.p_vel[active, 1] += GRAVITY * dt
            self.p_life[active] -= 2.0 * dt

        # Camera Collision Logic
        pivot_pos = np.array([px, py, pz], dtype='f4')
        cam_dir_x = np.cos(self.cam_yaw) * np.cos(self.cam_pitch)
        cam_dir_y = np.sin(self.cam_pitch)
        cam_dir_z = np.sin(self.cam_yaw) * np.cos(self.cam_pitch)
        ray_dir = -np.array([cam_dir_x, cam_dir_y, cam_dir_z], dtype='f4')

        closest_hit = self.target_zoom
        WALL_MARGIN = 0.2
        for p in self.platforms:
            b_min = np.array([p.x - p.hw, p.y - p.hh, p.z - p.hd])
            b_max = np.array([p.x + p.hw, p.y + p.hh, p.z + p.hd])
            dist = ray_aabb_intersect(pivot_pos, ray_dir, b_min, b_max)
            if dist > 0.0 and (dist - WALL_MARGIN) < closest_hit:
                closest_hit = dist - WALL_MARGIN

        self.current_zoom = max(0.4, closest_hit)
        return px, py, pz

    def check_overlap(self, x, y, z, p):
        return not (
                    x + self.pw <= p.x - p.hw or x - self.pw >= p.x + p.hw or y + self.ph <= p.y - p.hh or y - self.ph >= p.y + p.hh or z + self.pd <= p.z - p.hd or z - self.pd >= p.z + p.hd)

# --- 4. RENDERER ---
class Renderer:
    def __init__(self, ctx, platforms):
        self.ctx = ctx
        self.platforms = platforms
        self.image = ctx.image(WINDOW_SIZE, 'rgba8unorm')
        self.depth = ctx.image(WINDOW_SIZE, 'depth24plus')
        self.vbo = ctx.buffer(cube_vertices)
        self.instance_buffer = ctx.buffer(size=2000 * 16)

        # Buffers
        self.particle_staging = np.zeros((2000, 4), dtype='f4')
        self.proj_buf = np.eye(4, dtype='f4')
        self.view_buf = np.eye(4, dtype='f4')
        self.view_proj_buf = np.eye(4, dtype='f4')
        self.model_buf = np.eye(4, dtype='f4')
        self.mvp_buf = np.eye(4, dtype='f4')
        self.color_buf = np.array([1.0, 1.0, 1.0], dtype='f4')
        self.selected_buf = np.array([0], dtype='i4')  # For GLSL 450 boolean
        self.time_buf = np.array([0.0], dtype='f4')

        # Shared UBO
        self.camera_ubo = ctx.buffer(size=128)
        self.camera_ubo_staging = np.empty(32, dtype='f4')

        # 3. Main 3D Pipeline
        self.pipeline_3d = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout(location = 0) in vec3 in_vert;

                layout(std140, binding = 0) uniform Camera {
                    mat4 view;
                    mat4 proj;
                };

                layout(location = 0) uniform mat4 mvp;
                layout(location = 1) uniform mat4 model;
                layout(location = 2) uniform vec3 color;

                layout(location = 0) out vec3 v_color;
                layout(location = 1) out vec3 v_world_pos;
                layout(location = 2) out vec3 v_local_pos; // Pass local for edge detection

                void main() {
                    v_color = color;
                    v_world_pos = (model * vec4(in_vert, 1.0)).xyz;
                    v_local_pos = in_vert; 
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 450 core
                layout(location = 0) in vec3 v_color;
                layout(location = 1) in vec3 v_world_pos;
                layout(location = 2) in vec3 v_local_pos;

                layout(location = 3) uniform int is_selected; 
                layout(location = 4) uniform float u_time;

                layout(location = 0) out vec4 out_color;

                void main() {
                    vec3 normal = normalize(cross(dFdx(v_world_pos), dFdy(v_world_pos)));
                    vec3 sun_dir = normalize(vec3(0.5, 0.8, 0.3));
                    float diff = max(dot(normal, sun_dir), 0.0);
                    vec3 ambient = vec3(0.2, 0.2, 0.3);
                    vec3 final_color = v_color * (diff + 0.5) + v_color * ambient;

                    // --- HIGHLIGHT LOGIC ---
                    if (is_selected == 1) {
                        // 1. Pulsing Body
                        float pulse = (sin(u_time * 5.0) * 0.5 + 0.5) * 0.3; 
                        final_color += vec3(0.2, 0.2, 0.0) + (vec3(0.8, 0.8, 0.0) * pulse);

                        // 2. Wireframe/Edge Detection (Box Topology)
                        // Local pos is -0.5 to 0.5. Check if we are near +/- 0.48
                        vec3 a = step(0.48, abs(v_local_pos)); 
                        // If 2 components are near edge, it's a line. If 3, it's a corner.
                        float is_edge = max(max(a.x * a.y, a.y * a.z), a.x * a.z);

                        if (is_edge > 0.0) {
                            final_color = vec3(1.0, 0.9, 0.2); // Bright Yellow Edge
                        }
                    }
                    out_color = vec4(final_color, 1.0);
                }
            ''',
            layout=[{'name': 'Camera', 'binding': 0}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': self.camera_ubo}],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            depth={'func': 'less', 'write': True},
            vertex_buffers=[*zengl.bind(self.vbo, '3f', 0), ],
            uniforms={
                'mvp': [0.0] * 16,
                'model': [0.0] * 16,
                'color': [1.0, 1.0, 1.0],
                'is_selected': 0,
                'u_time': 0.0
            },
            vertex_count=36,
        )

        # 4. Particle Pipeline
        self.pipeline_particles = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout(location = 0) in vec3 in_vert;
                layout(location = 1) in vec3 in_pos; 
                layout(location = 2) in float in_scale;

                layout(std140, binding = 0) uniform Camera {
                    mat4 view;
                    mat4 proj;
                };

                layout(location = 0) out vec3 v_color;

                void main() {
                    vec3 world_pos = in_pos + in_vert * in_scale;
                    gl_Position = proj * view * vec4(world_pos, 1.0);
                    float life = in_scale / 0.15; 
                    v_color = vec3(0.7, 0.7, 0.8) * life;
                }
            ''',
            fragment_shader='''
                #version 450 core
                out vec4 out_color;
                in vec3 v_color;
                void main() { out_color = vec4(v_color, 1.0); }
            ''',
            layout=[{'name': 'Camera', 'binding': 0}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': self.camera_ubo}],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            depth={'func': 'less', 'write': False},
            vertex_buffers=[
                *zengl.bind(self.vbo, '3f', 0),
                *zengl.bind(self.instance_buffer, '3f 1f /i', 1, 2)
            ],
            uniforms={},
            vertex_count=36,
            instance_count=0,
        )

        self.pipeline_grid = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout(location = 0) in vec3 in_vert;

                layout(std140, binding = 0) uniform Camera {
                    mat4 view;
                    mat4 proj;
                };
                uniform mat4 model;

                out vec3 v_world_pos;

                void main() {
                    vec4 world = model * vec4(in_vert, 1.0);
                    v_world_pos = world.xyz;
                    gl_Position = proj * view * world;
                }
            ''',
            fragment_shader='''
                #version 450 core
                in vec3 v_world_pos;
                out vec4 out_color;

                void main() {
                    // Anti-aliased Grid Logic
                    vec2 coord = v_world_pos.xz;
                    vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
                    float line = min(grid.x, grid.y);
                    float alpha = 1.0 - min(line, 1.0);

                    // Thicker lines every 10 units
                    vec2 grid10 = abs(fract(coord * 0.1 - 0.5) - 0.5) / fwidth(coord * 0.1);
                    float line10 = min(grid10.x, grid10.y);
                    float alpha10 = 1.0 - min(line10, 1.0);

                    vec3 color = vec3(0.6); // Grid color
                    float final_alpha = max(alpha * 0.3, alpha10 * 0.8);

                    // Fade out distance
                    float dist = length(v_world_pos.xz); // Distance from center
                    final_alpha *= max(0.0, 1.0 - dist / 50.0);

                    if (final_alpha <= 0.0) discard;
                    out_color = vec4(color, final_alpha);
                }
            ''',
            layout=[{'name': 'Camera', 'binding': 0}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': self.camera_ubo}],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            # Grid is transparent, read depth but don't write
            depth={'func': 'less', 'write': False},
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            vertex_buffers=[*zengl.bind(self.vbo, '3f', 0)],
            uniforms={'model': [0.0] * 16},
            vertex_count=36,
        )

        self.cam_x, self.cam_y, self.cam_z = 0, 0, 0
        self.global_time = 0.0

    def _render_obj(self, r, g, b, selected=False):
        self.pipeline_3d.uniforms['mvp'][:] = memoryview(self.mvp_buf).cast('B')
        self.pipeline_3d.uniforms['model'][:] = memoryview(self.model_buf).cast('B')
        self.color_buf[:] = [r, g, b]
        self.pipeline_3d.uniforms['color'][:] = memoryview(self.color_buf).cast('B')
        self.selected_buf[0] = 1 if selected else 0
        self.pipeline_3d.uniforms['is_selected'][:] = memoryview(self.selected_buf).cast('B')
        self.pipeline_3d.uniforms['u_time'][:] = memoryview(self.time_buf).cast('B')
        self.pipeline_3d.render()

    def draw(self, cam_pos, cam_yaw, cam_pitch, player_pos=None, selected_idx=-1, time_now=0.0, is_edit_mode=False):
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        self.global_time = time_now
        self.time_buf[0] = time_now

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        get_perspective(60.0, aspect, 0.1, 100.0, out=self.proj_buf)

        # Calculate LookAt
        tx = cam_pos[0] + np.cos(cam_yaw) * np.cos(cam_pitch)
        ty = cam_pos[1] + np.sin(cam_pitch)
        tz = cam_pos[2] + np.sin(cam_yaw) * np.cos(cam_pitch)

        get_lookat(
            np.array(cam_pos, 'f4'),
            np.array([tx, ty, tz], 'f4'),
            np.array([0, 1, 0], 'f4'),
            out=self.view_buf
        )

        self.camera_ubo_staging[:16] = self.view_buf.flatten()
        self.camera_ubo_staging[16:] = self.proj_buf.flatten()
        self.camera_ubo.write(self.camera_ubo_staging)

        np.matmul(self.view_buf, self.proj_buf, out=self.view_proj_buf)

        # Draw Platforms
        for i, p in enumerate(self.platforms):
            get_model_matrix(self.model_buf, p.x, p.y, p.z, p.hw * 2, p.hh * 2, p.hd * 2)
            np.matmul(self.model_buf, self.view_proj_buf, out=self.mvp_buf)
            self._render_obj(0.3, 0.7, 0.4, selected=(i == selected_idx))

        # Draw Player
        if not is_edit_mode and player_pos is not None:
            px, py, pz = player_pos
            get_model_matrix(self.model_buf, px, py, pz, 0.4, 0.4, 0.4)
            np.matmul(self.model_buf, self.view_proj_buf, out=self.mvp_buf)
            self._render_obj(1.0, 0.5, 0.0)

        # Draw Grid (Only in Edit Mode)
        if is_edit_mode:
            # Scale cube to be a large flat plane at y=0 (or slightly below platforms)
            # Size 100x100, very thin Y
            get_model_matrix(self.model_buf, 0.0, -0.05, 0.0, 100.0, 0.01, 100.0)
            self.pipeline_grid.uniforms['model'][:] = memoryview(self.model_buf).cast('B')
            self.pipeline_grid.render()

        # Draw Particles
        active_indices = physics.p_life > 0
        count = np.sum(active_indices)
        if count > 0:
            self.particle_staging[:count, 0:3] = physics.p_pos[active_indices]
            self.particle_staging[:count, 3] = physics.p_life[active_indices] * 0.15
            self.instance_buffer.write(offset=0, data=self.particle_staging[:count])
            self.pipeline_particles.instance_count = int(count)
            self.pipeline_particles.render()

        self.image.blit()
        self.ctx.end_frame()


# --- 6. MAIN APP LOGIC ---
class GameState:
    def __init__(self):
        self.mode = "PLAY"  # or "EDIT"

class Game:
    TARGET_FPS = 60.0
    FIXED_DT = 1.0 / TARGET_FPS
    SUB_STEPS = 4
    PHYSICS_DT = FIXED_DT / SUB_STEPS
    def __init__(self, window, app_state, renderer, physics, editor, platforms):
        self.window = window
        self.renderer = renderer
        self.physics = physics
        self.editor = editor
        self.app_state = app_state
        self.platforms = platforms

    @staticmethod
    def get_initial_platforms():
        level = Path("platformer3d") / "level.json"
        if level.exists():
            with open(level, "r") as f:
                data = json.load(f)
                return [Platform(**d) for d in data]
        return [Platform(0.0, -1.0, 0.0, 4.0, 0.2, 4.0)]

    def load_level(self):
        level = Path("platformer3d") / "level.json"
        if level.exists():
            with open(level, "r") as f:
                data = json.load(f)
                for d in data: self.platforms.append(Platform(**d))
        else:
            self.platforms = [Platform(0.0, -1.0, 0.0, 4.0, 0.2, 4.0)]

    def on_key(self, win, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True) # add menu soon?

        # Toggle Mode
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            self.app_state.mode = "EDIT" if self.app_state.mode == "PLAY" else "PLAY"
            print(f"Switched to {self.app_state.mode} mode")
            # Sync Camera positions
            if self.app_state.mode == "EDIT":
                # Calculate where the physics camera was
                zoom = self.physics.current_zoom
                px, py, pz = self.physics.player_data[0:3]
                yaw, pitch = self.physics.cam_yaw, self.physics.cam_pitch
                self.editor.cam_x = px - np.cos(yaw) * np.cos(pitch) * zoom
                self.editor.cam_y = py - np.sin(pitch) * zoom
                self.editor.cam_z = pz - np.sin(yaw) * np.cos(pitch) * zoom
                self.editor.cam_yaw = yaw
                self.editor.cam_pitch = pitch
            else:
                # Sync physics yaw/pitch (approximate)
                self.physics.cam_yaw = self.editor.cam_yaw
                self.physics.cam_pitch = self.editor.cam_pitch

        # Input Routing
        if self.app_state.mode == "PLAY":
            if action == glfw.PRESS:
                self.physics.keys[key] = True
            elif action == glfw.RELEASE:
                self.physics.keys[key] = False
        else:
            if action == glfw.PRESS:
                self.editor.keys[key] = True
            elif action == glfw.RELEASE:
                self.editor.keys[key] = False
            self.editor.handle_input(key, action, mods)

    def on_mouse(self, win, xpos, ypos):
        if not hasattr(self, '_last_mouse'):
            self._last_mouse = (xpos, ypos)

        dx = xpos - self._last_mouse[0]
        dy = ypos - self._last_mouse[1]
        self._last_mouse = (xpos, ypos)

        if self.app_state.mode == "PLAY":
            self.physics.cam_yaw += dx * MOUSE_SENSITIVITY
            self.physics.cam_pitch -= dy * MOUSE_SENSITIVITY
            self.physics.cam_pitch = max(-1.5, min(1.5, self.physics.cam_pitch))
        else:
            self.editor.cam_yaw += dx * MOUSE_SENSITIVITY
            self.editor.cam_pitch -= dy * MOUSE_SENSITIVITY
            self.editor.cam_pitch = max(-1.5, min(1.5, self.editor.cam_pitch))

    def on_mouse_click(self, win, button, action, mods):
        if self.app_state.mode == "EDIT" and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.editor.select_object()

    def on_scroll(self, win, xoff, yoff):
        if self.app_state.mode == "PLAY":
            self.physics.target_zoom -= yoff * SCROLL_SENSITIVITY
            self.physics.target_zoom = max(1.0, min(15.0, self.physics.target_zoom))

    def setup_callbacks(self):
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_click)
        glfw.set_scroll_callback(self.window, self.on_scroll)

    def run(self):
        accumulator = 0.0
        last_time = time.perf_counter()

        px, py, pz = self.physics.player_data[0:3]
        prev_px, prev_py, prev_pz = px, py, pz

        while not glfw.window_should_close(self.window):
            current_time = time.perf_counter()
            frame_time = current_time - last_time
            last_time = current_time
            if frame_time > 0.25: frame_time = 0.25

            # Logic Update
            if self.app_state.mode == "PLAY":
                accumulator += frame_time
                while accumulator >= self.FIXED_DT:
                    prev_px, prev_py, prev_pz = px, py, pz
                    for _ in range(self.SUB_STEPS):
                        px, py, pz = self.physics.step(self.PHYSICS_DT)
                    accumulator -= self.FIXED_DT
                alpha = accumulator / self.FIXED_DT

                # Interpolated Player Position
                ix = prev_px * (1.0 - alpha) + px * alpha
                iy = prev_py * (1.0 - alpha) + py * alpha
                iz = prev_pz * (1.0 - alpha) + pz * alpha

                # Calculate Follow Cam
                target_cam_x = ix - np.cos(self.physics.cam_yaw) * np.cos(self.physics.cam_pitch) * self.physics.current_zoom
                target_cam_y = iy - np.sin(self.physics.cam_pitch) * self.physics.current_zoom
                target_cam_z = iz - np.sin(self.physics.cam_yaw) * np.cos(self.physics.cam_pitch) * self.physics.current_zoom

                # Simple lerp for camera

                # Smooth factor independent of frame rate
                # Adjust '18.0' to change tightness (higher = tighter)
                cam_lerp_config = 18.0
                cam_lerp = 1.0 - np.exp(-cam_lerp_config * frame_time)

                self.renderer.cam_x += (target_cam_x - self.renderer.cam_x) * cam_lerp
                self.renderer.cam_y += (target_cam_y - self.renderer.cam_y) * cam_lerp
                self.renderer.cam_z += (target_cam_z - self.renderer.cam_z) * cam_lerp

                self.renderer.draw(
                    cam_pos=(self.renderer.cam_x, self.renderer.cam_y, self.renderer.cam_z),
                    cam_yaw=self.physics.cam_yaw,
                    cam_pitch=self.physics.cam_pitch,
                    player_pos=(ix, iy, iz),  # <--- The smooth coordinates
                    selected_idx=-1,
                    time_now=current_time,
                    is_edit_mode=False
                )

            else:  # EDIT MODE
                self.editor.update(frame_time)
                self.renderer.draw(
                    cam_pos=(self.editor.cam_x, self.editor.cam_y, self.editor.cam_z),
                    cam_yaw=self.editor.cam_yaw,
                    cam_pitch=self.editor.cam_pitch,
                    selected_idx=self.editor.selected_index,
                    time_now=current_time,
                    is_edit_mode=True
                )

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def bootstrap_ogl():
    """Handles all the 'boring' hardware/driver initialization."""
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WINDOW_SIZE[0], WINDOW_SIZE[1], "ZenGL 3D Platformer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create Window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    if glfw.raw_mouse_motion_supported():
        glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)

    return window, zengl.context()

if __name__ == "__main__":
    window, ctx = bootstrap_ogl()

    # 2. Data and State
    app_state = GameState()
    platforms = Game.get_initial_platforms()

    # 3. Initialize subsystems
    renderer = Renderer(ctx, platforms)
    physics = PhysicsEngine(platforms)
    editor = Editor(platforms, Platform, ray_aabb_intersect)

    # 4. Create the game, passing the already-created window
    game = Game(
        window=window,
        app_state=app_state,
        renderer=renderer,
        physics=physics,
        editor=editor,
        platforms=platforms
    )

    game.setup_callbacks()
    game.run()