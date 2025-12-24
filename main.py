from typing import TYPE_CHECKING
if TYPE_CHECKING: from zengl import BlendSettings

import glfw
import zengl
import numpy as np
import sys
import threading
import time

# 1. Initialize Window & Context
if not glfw.init():
    sys.exit()

# Verify Free-Threading status
is_free_threaded = hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() == False
GIL_STATE = "No GIL (True Parallelism)" if is_free_threaded else "GIL Active"
print(f"Python {sys.version.split()[0]} | {GIL_STATE}")
gil_tpl = f"3.14t Swarm | {GIL_STATE}"

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

# Disable VSync to see true physics throughput
glfw.window_hint(glfw.DOUBLEBUFFER, True)
window = glfw.create_window(1280, 720, gil_tpl, None, None)
glfw.make_context_current(window)
glfw.swap_interval(0)

ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

# 2. CONFIGURATIONS
# Increased count because 3.14t can handle it
TRIANGLE_COUNT = 200000
NUM_WORKERS = 8

# Physics Constants
MAX_SPEED = 12.0
FRICTION = 0.99
REPULSE_FORCE = 1200.0
ATTRACT_FORCE = 40.0

# Shared Arrays (NumPy 2.x releases the GIL on heavy ops, but here we are in Python loop)
props = np.random.uniform(0.8, 1.2, (TRIANGLE_COUNT, 4)).astype('f4')
target = np.array([0.0, 0.0], dtype='f4')

# [x, y, vx, vy, kinetic, potential]
gpu_data = np.zeros((TRIANGLE_COUNT, 6), dtype='f4')

# Init positions
gpu_data[:, 0] = np.random.uniform(-600, 600, TRIANGLE_COUNT)
gpu_data[:, 1] = np.random.uniform(-350, 350, TRIANGLE_COUNT)
gpu_data[:, 2:4] = np.random.uniform(-1, 1, (TRIANGLE_COUNT, 2))

running = True
is_repelling = False


m_prev_x, m_prev_y = glfw.get_cursor_pos(window)
mouse_vel = np.array([0.0, 0.0], dtype='f4')


# 3. WORKER LOGIC (Free-Threaded)
def worker_logic(start_idx, end_idx):
    my_data = gpu_data[start_idx:end_idx]
    my_props = props[start_idx:end_idx]

    # Pre-allocate slice views
    pos_x, pos_y = my_data[:, 0], my_data[:, 1]
    vel_x, vel_y = my_data[:, 2], my_data[:, 3]
    dt = 0.016

    while running:
        # 1. Global Sync Pulse
        t = time.perf_counter()
        pulse = (np.sin(t * 4.0) * 0.4) + 1.0

        # 2. Distance Math
        dx, dy = target[0] - pos_x, target[1] - pos_y
        dist_sq = dx * dx + dy * dy + 60.0  # Slightly higher softening
        inv_dist = 1.0 / np.sqrt(dist_sq)

        # 3. Force Calculation
        if is_repelling:
            # SHOCKWAVE: High-power inverse cubic
            force = -REPULSE_FORCE * (inv_dist ** 3) * 12000.0
            # Add chaotic turbulence and slant it with mouse movement
            ax = (dx * force + mouse_vel[0] * 8.0 + np.random.uniform(-10, 10, len(dx))) / my_props[:, 0]
            ay = (dy * force + mouse_vel[1] * 8.0 + np.random.uniform(-10, 10, len(dy))) / my_props[:, 0]
        else:
            # RIVER: Pulsing attraction + Mouse Wind
            force = (ATTRACT_FORCE * pulse) * inv_dist
            wind_str = 25.0 * (inv_dist ** 1.5)

            ax = (dx * force + mouse_vel[0] * wind_str) / my_props[:, 0]
            ay = (dy * force + mouse_vel[1] * wind_str) / my_props[:, 0]

            # SWIRL
            swirl_mag = 22.0 * inv_dist
            ax -= (dy * swirl_mag) / my_props[:, 0]
            ay += (dx * swirl_mag) / my_props[:, 0]

        # 4. Integration (Semi-Implicit)
        vel_x += ax * dt
        vel_y += ay * dt

        # Apply Friction
        vel_x *= FRICTION
        vel_y *= FRICTION

        # Hard Speed Cap
        speed_sq = vel_x * vel_x + vel_y * vel_y
        over_limit = speed_sq > (MAX_SPEED * MAX_SPEED)
        if np.any(over_limit):
            scale = MAX_SPEED / np.sqrt(speed_sq[over_limit])
            vel_x[over_limit] *= scale
            vel_y[over_limit] *= scale

        # Position Update
        pos_x += vel_x
        pos_y += vel_y

        # Screen Wrap
        np.putmask(pos_x, pos_x > 650, -650)
        np.putmask(pos_x, pos_x < -650, 650)
        np.putmask(pos_y, pos_y > 370, -370)
        np.putmask(pos_y, pos_y < -370, 370)

        # Telemetry for Shader (speed squared normalized)
        my_data[:, 4] = speed_sq * 0.012
        elapsed = time.perf_counter() - t
        sleep_time = dt - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time) # So that your CPU doesn't cry when you try to run 200000 triangles


# Start Workers
chunk = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    s = i * chunk
    e = TRIANGLE_COUNT if i == NUM_WORKERS - 1 else (i + 1) * chunk
    threading.Thread(target=worker_logic, args=(s, e), daemon=True).start()

# 4. PIPELINE
shape = ctx.buffer(np.array([0, 0.05, -0.02, -0.02, 0.02, -0.02], dtype='f4'))

vertex_shader = '''
    #version 450 core
    layout (location = 0) in vec2 in_vert;
    layout (location = 1) in vec4 in_inst;
    layout (location = 2) in vec2 in_energy;
    
    out float v_energy;
    
    void main() {
        vec2 pos = in_inst.xy;
        vec2 vel = in_inst.zw;
        float speed = length(vel);
        v_energy = clamp(in_energy.x, 0.0, 1.0);
    
        float angle = atan(vel.y, vel.x) - 1.5708;
        
        // EXCITING: Velocity Stretching
        // Fast triangles become long needles (1.0 base + 0.15 * speed)
        float stretch = 1.0 + (speed * 0.15);
        vec2 stretched_vert = in_vert * vec2(1.0, stretch);
    
        float c = cos(angle);
        float s = sin(angle);
        mat2 rot = mat2(c, s, -s, c);
    
        // Scale also grows slightly with energy
        float size_scale = 0.8 + (v_energy * 0.5);
        gl_Position = vec4((rot * stretched_vert * size_scale) + (pos / vec2(640.0, 360.0)), 0.0, 1.0);
    }
'''

fragment_shader = '''
    #version 450 core
    in float v_energy;
    layout (location = 0) out vec4 out_color;
    // Fragment Shader
    void main() {
        float e = pow(v_energy, 2.5);
        
        // palette: Deep Blue -> Electric Cyan -> White Hot -> Solar Orange
        vec3 color = mix(vec3(0.02, 0.05, 0.2), vec3(0.0, 0.8, 1.0), e);
        if (e > 0.7) {
            color = mix(color, vec3(2.0, 1.2, 0.5), (e - 0.7) * 3.3);
        }
        
        out_color = vec4(color, 1.0);
    }
'''

instance_buffs = [ctx.buffer(size=TRIANGLE_COUNT * 24) for _ in range(2)]

pipelines = [
    ctx.pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
        instance_count=TRIANGLE_COUNT,
        vertex_buffers=[
            *zengl.bind(shape, '2f', 0),
            *zengl.bind(instance_buffs[i], '4f 2f /i', 1, 2),
        ],
    ) for i in range(2)
]

fade_blend_settings : BlendSettings = {
    'enable': True,
    'src_color': 'src_alpha',
    'dst_color': 'one_minus_src_alpha'
}

# Trail effect
fade_pipe = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        vec2 v[4] = vec2[](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
        void main() { gl_Position = vec4(v[gl_VertexID], 0, 1); }
    ''',
    fragment_shader='''
        #version 450 core
        out vec4 c;
        void main() { c = vec4(0, 0, 0, 0.15); }
    ''',
    framebuffer=[image],
    topology='triangle_strip',
    vertex_count=4,
    blend=fade_blend_settings,
)

# 5. RENDER LOOP
frame = 0
t_prev = time.perf_counter()

while not glfw.window_should_close(window):
    mx, my = glfw.get_cursor_pos(window)
    # Calculate velocity: current - previous
    mouse_vel[0] = mx - m_prev_x
    mouse_vel[1] = (360 - my) - (360 - m_prev_y)  # Screen space Y is inverted
    m_prev_x, m_prev_y = mx, my
    is_repelling = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    target[0] = mx - 640
    target[1] = 360 - my

    ctx.new_frame()
    fade_pipe.render()

    idx = frame % 2
    # In 3.14t, this read is racy against the worker writes.
    # We accept the "motion blur" artifact for performance.
    instance_buffs[idx].write(gpu_data)
    pipelines[idx].render()

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

    frame += 1
    if frame % 60 == 0:
        t_now = time.perf_counter()
        fps = 60 / (t_now - t_prev)
        t_prev = t_now
        glfw.set_window_title(window, f"{gil_tpl} | FPS: {int(fps)} | Particles: {TRIANGLE_COUNT}")

glfw.terminate()