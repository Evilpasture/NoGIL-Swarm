import glfw
import zengl
import numpy as np
import trimesh
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
            self.p_idx = (self.p_idx + 1) % self.max_particles

    def step(self, dt):
        px, py, pz, vx, vy, vz = self.player_data

        # --- PHYSICS CONSTANTS ---
        GRAVITY = -30.0
        JUMP_FORCE = 10.0

        # MOVEMENT TUNING
        GROUND_MAX_SPEED = 6.0
        GROUND_ACCEL = 50.0
        GROUND_FRICTION = 15.0

        AIR_MAX_SPEED = 6.0
        AIR_ACCEL = 20.0
        AIR_FRICTION = 0.5

        # 1. Select Parameters
        if self.on_ground:
            max_speed = GROUND_MAX_SPEED
            accel = GROUND_ACCEL
            friction = GROUND_FRICTION
        else:
            max_speed = AIR_MAX_SPEED
            accel = AIR_ACCEL
            friction = AIR_FRICTION

        # 2. Gravity
        vy += GRAVITY * dt
        vy = max(vy, -30.0)

        # 3. Friction
        damping = 1.0 / (1.0 + friction * dt)
        vx *= damping
        vz *= damping

        # 4. Input
        for_x, for_z = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        right_x, right_z = np.cos(self.cam_yaw + np.pi / 2), np.sin(self.cam_yaw + np.pi / 2)
        ix, iz = 0.0, 0.0
        if self.keys.get(glfw.KEY_W): ix += for_x; iz += for_z
        if self.keys.get(glfw.KEY_S): ix -= for_x; iz -= for_z
        if self.keys.get(glfw.KEY_A): ix -= right_x; iz -= right_z
        if self.keys.get(glfw.KEY_D): ix += right_x; iz += right_z

        # 5. Vector Projection Movement
        if ix != 0 or iz != 0:
            l = np.sqrt(ix * ix + iz * iz)
            wish_dir_x = ix / l
            wish_dir_z = iz / l
            current_speed_in_wish_dir = vx * wish_dir_x + vz * wish_dir_z
            add_speed = max_speed - current_speed_in_wish_dir
            if add_speed > 0:
                accel_speed = min(accel * dt, add_speed)
                vx += wish_dir_x * accel_speed
                vz += wish_dir_z * accel_speed

        # Jump
        if self.keys.get(glfw.KEY_SPACE) and self.on_ground:
            self.spawn_burst(px, py - 0.1, pz, 20)
            vy = JUMP_FORCE
            self.on_ground = False

        # --- COLLISION LOGIC ---
        VERTICAL_ALLOWANCE = 0.25

        # X-Axis
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

        # Z-Axis
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

        # Y-Axis
        # prev_py = py
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

        # Particles
        active = self.p_life > 0
        if np.any(active):
            self.p_pos[active] += self.p_vel[active] * dt
            self.p_vel[active, 1] += GRAVITY * dt
            self.p_life[active] -= 2.0 * dt

        # --- CAMERA LOGIC ---
        pivot_pos = np.array([px, py, pz], dtype='f4')
        cam_dir_x = np.cos(self.cam_yaw) * np.cos(self.cam_pitch)
        cam_dir_y = np.sin(self.cam_pitch)
        cam_dir_z = np.sin(self.cam_yaw) * np.cos(self.cam_pitch)
        ray_dir = -np.array([cam_dir_x, cam_dir_y, cam_dir_z], dtype='f4')

        closest_hit = self.target_zoom

        CAM_RADIUS = 0.25
        PLAYER_COLLISION_RADIUS = self.pw

        # FIX: Added a hard margin so we don't look directly at the polygon face
        WALL_MARGIN = 0.2

        for p in self.platforms:
            b_min_hard = np.array([p.x - p.hw, p.y - p.hh, p.z - p.hd])
            b_max_hard = np.array([p.x + p.hw, p.y + p.hh, p.z + p.hd])

            # 1. Hard Collision (Actual Geometry)
            dist_hard = ray_aabb_intersect(pivot_pos, ray_dir, b_min_hard, b_max_hard)

            if dist_hard > 0.0:
                # Apply margin to pull camera back from the wall surface
                safe_dist = dist_hard - WALL_MARGIN
                if safe_dist < closest_hit:
                    closest_hit = safe_dist

            # 2. Soft Collision (Sphere Proxy)
            b_min_soft = b_min_hard - CAM_RADIUS
            b_max_soft = b_max_hard + CAM_RADIUS
            dist_soft = ray_aabb_intersect(pivot_pos, ray_dir, b_min_soft, b_max_soft)

            if 0.0 < dist_soft < closest_hit:
                hit_point = pivot_pos + ray_dir * dist_soft
                dist_to_player = np.linalg.norm(hit_point - pivot_pos)
                if dist_to_player >= CAM_RADIUS + PLAYER_COLLISION_RADIUS:
                    closest_hit = dist_soft

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

        # 1. Geometry (The Cube)
        self.vbo = ctx.buffer(cube_vertices)

        # 2. Instance Buffer (Holds data for ALL particles)
        # Format: 3 floats (pos) + 1 float (scale) = 16 bytes per particle
        self.instance_buffer = ctx.buffer(size=2000 * 16)

        # --- PRE-ALLOCATED BUFFERS (The Anti-Leak Mechanism) ---
        # We create these ONCE. We never use np.empty/np.zeros in draw() again.

        self.particle_staging = np.zeros((2000, 4), dtype='f4')

        self.proj_buf = np.eye(4, dtype='f4')
        self.view_buf = np.eye(4, dtype='f4')
        self.view_proj_buf = np.eye(4, dtype='f4')
        self.model_buf = np.eye(4, dtype='f4')
        self.mvp_buf = np.eye(4, dtype='f4')

        self.color_buf = np.array([1.0, 1.0, 1.0], dtype='f4')

        # Holds 2 matrices: View (64 bytes) + Proj (64 bytes) = 128 bytes
        # Binding = 0
        self.camera_ubo = ctx.buffer(size=128)

        self.camera_ubo_staging = np.empty(32, dtype='f4') # 32 floats = 128 bytes


        # Shadow? for later, leaving the commented out code for now
        self.shadow_map = ctx.image((2048, 2048), 'depth24plus')

        # pipeline_shadow = ctx.pipeline(
        #     vertex_shader='''
        #         #version 330 core
        #         layout (location = 0) in vec3 in_vert;
        #         uniform mat4 light_mvp;
        #         void main() {
        #             gl_Position = light_mvp * vec4(in_vert, 1.0);
        #         }
        #     ''',
        #     fragment_shader='''
        #         #version 330 core
        #         // stub
        #         void main() {
        #         // stub
        #         }
        #     ''',
        #     framebuffer=[self.shadow_map],  # No color attachment, depth only
        #     topology='triangles',
        #     depth={'func': 'less', 'write': True},
        #     vertex_buffers=[*zengl.bind(self.vbo, '3f', 0)],
        #     uniforms={'mvp': [0.0] * 16, 'model': [0.0] * 16, 'color': [1.0, 1.0, 1.0]},
        #     vertex_count=36,
        # )

        # 3. Main 3D Pipeline (For Player & Platforms)
        self.pipeline_3d = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout(location = 0) in vec3 in_vert;

                // 1. Shared Data via UBO (Binding 0)
                layout(std140, binding = 0) uniform Camera {
                    mat4 view;
                    mat4 proj;
                };

                // 2. Per-Object Data via Standard Uniforms
                // These change every draw call, so they are faster as standard uniforms
                layout(location = 0) uniform mat4 mvp;
                layout(location = 1) uniform mat4 model;
                layout(location = 2) uniform vec3 color;

                layout(location = 0) out vec3 v_color;
                layout(location = 1) out vec3 v_world_pos;

                void main() {
                    v_color = color;
                    v_world_pos = (model * vec4(in_vert, 1.0)).xyz;
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 450 core
                layout(location = 0) in vec3 v_color;
                layout(location = 1) in vec3 v_world_pos;
                layout(location = 0) out vec4 out_color;
                void main() {
                    vec3 normal = normalize(cross(dFdx(v_world_pos), dFdy(v_world_pos)));
                    vec3 sun_dir = normalize(vec3(0.5, 0.8, 0.3));
                    float diff = max(dot(normal, sun_dir), 0.0);
                    vec3 ambient = vec3(0.2, 0.2, 0.3);
                    out_color = vec4(v_color * (diff + 0.5) + v_color * ambient, 1.0);
                }
            ''',
            layout=[
                {
                    'name': 'Camera',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.camera_ubo,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            depth={'func': 'less', 'write': True},
            vertex_buffers=[*zengl.bind(self.vbo, '3f', 0), ],
            # Note: We bind the UBO globally in draw(), not here in 'uniforms' dict
            # We only define standard uniforms here
            uniforms={
                'mvp': [0.0] * 16,
                'model': [0.0] * 16,
                'color': [1.0, 1.0, 1.0]
            },
            vertex_count=36,
        )

        # 4. Particle Pipeline (INSTANCED)
        self.pipeline_particles = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout(location = 0) in vec3 in_vert;
                layout(location = 1) in vec3 in_pos; 
                layout(location = 2) in float in_scale;

                // UBO Binding 0 (Shared with Main Pipeline!)
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
            layout=[
                {
                    'name': 'Camera',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': self.camera_ubo,
                },
            ],
            framebuffer=[self.image, self.depth],
            topology='triangles',
            depth={'func': 'less', 'write': False},  # Particles usually don't write depth (optional)
            vertex_buffers=[
                *zengl.bind(self.vbo, '3f', 0),  # The Cube Mesh
                *zengl.bind(self.instance_buffer, '3f 1f /i', 1, 2)  # The Particle Data (/i = per instance)
            ],
            uniforms={},
            vertex_count=36,
            instance_count=0,
        )

        self.cam_x, self.cam_y, self.cam_z = 0, 3, 5

    def _render_obj(self, r, g, b):
        self.pipeline_3d.uniforms['mvp'][:] = memoryview(self.mvp_buf).cast('B')
        self.pipeline_3d.uniforms['model'][:] = memoryview(self.model_buf).cast('B')
        self.color_buf[0] = r
        self.color_buf[1] = g
        self.color_buf[2] = b
        self.pipeline_3d.uniforms['color'][:] = memoryview(self.color_buf).cast('B')
        self.pipeline_3d.render()

    def draw(self, px, py, pz, cam_yaw, cam_pitch, zoom_dist, dt):
        self.ctx.new_frame()
        self.image.clear()
        self.depth.clear()

        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        get_perspective(60.0, aspect, 0.1, 100.0, out=self.proj_buf)


        # Camera Smooth Follow (could change px, py, pz to rx, ry, rz... maybe)
        target_cam_x = px - np.cos(cam_yaw) * np.cos(cam_pitch) * zoom_dist
        target_cam_y = py - np.sin(cam_pitch) * zoom_dist
        target_cam_z = pz - np.sin(cam_yaw) * np.cos(cam_pitch) * zoom_dist

        self.cam_x += (target_cam_x - self.cam_x) * 0.1
        self.cam_y += (target_cam_y - self.cam_y) * 0.1
        self.cam_z += (target_cam_z - self.cam_z) * 0.1

        get_lookat(
            np.array([self.cam_x, self.cam_y, self.cam_z], 'f4'),
            np.array([px, py + 0.5, pz], 'f4'),
            np.array([0, 1, 0], 'f4'),
            out=self.view_buf
        )

        # We assume the layout in the shader is "view" then "proj"
        # Since buffers are Column-Major, we upload them directly.
        # Pack data into staging array: [View (16 floats)] + [Proj (16 floats)]
        self.camera_ubo_staging[:16] = self.view_buf.flatten()
        self.camera_ubo_staging[16:] = self.proj_buf.flatten()

        # Write to GPU once per frame
        self.camera_ubo.write(self.camera_ubo_staging)

        # Since buffers are Column-Major (A^T, B^T),
        # (P * V)^T = V^T * P^T
        # So we mul View_Buf @ Proj_Buf
        np.matmul(self.view_buf, self.proj_buf, out=self.view_proj_buf)

        # --- DRAW PLATFORMS ---
        for p in self.platforms:
            get_model_matrix(self.model_buf, p.x, p.y, p.z, p.hw * 2, p.hh * 2, p.hd * 2)
            # (P * V * M)^T = M^T * (V^T * P^T)
            np.matmul(self.model_buf, self.view_proj_buf, out=self.mvp_buf)
            self._render_obj(0.3, 0.7, 0.4)

            # 3. Render
            self._render_obj(0.3, 0.7, 0.4)

        # Draw player
        get_model_matrix(self.model_buf, px, py, pz, 0.4, 0.4, 0.4)
        np.matmul(self.model_buf, self.view_proj_buf, out=self.mvp_buf)
        self._render_obj(1.0, 0.5, 0.0)

        # --- OPTIMIZED PARTICLE DRAW ---
        # 1. Filter active particles
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


if __name__ == "__main__":
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
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