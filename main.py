from typing import TYPE_CHECKING
if TYPE_CHECKING: from zengl import BlendSettings

import glfw
import zengl
import numpy as np
import sys
import threading
import time
import gc

from zengl_extras import init

init(debug=True, gpu=False, dpi_aware=False, high_performance=False, opengl_core=False) # ignore this, just extra util

gc.disable()

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
TRIANGLE_COUNT = 3000000
NUM_WORKERS = 8
DT = 0.016

# Physics Constants
MAX_SPEED = 12.0
FRICTION = 0.99
REPULSE_FORCE = 1200.0
ATTRACT_FORCE = 40.0

buffers = [
    {
        'pos': ctx.buffer(size=TRIANGLE_COUNT * 8),
        'vel': ctx.buffer(size=TRIANGLE_COUNT * 8),
    } for _ in range(2)
]

props = np.random.uniform(0.8, 1.2, TRIANGLE_COUNT).astype('f4')
target = np.array([0.0, 0.0], dtype='f4')

# We use two separate sets of NumPy arrays to prevent "tearing"
# while the main thread is uploading
physics_data = {
    'pos': np.random.uniform(-600, 600, (TRIANGLE_COUNT, 2)).astype('f4'),
    'vel': np.random.uniform(-1, 1, (TRIANGLE_COUNT, 2)).astype('f4'),
}

running = True
is_repelling = False


# SYNC: One event to release workers, one barrier to catch them
start_sim = threading.Event()
done_sim = threading.Barrier(NUM_WORKERS + 1)

m_prev_x, m_prev_y = glfw.get_cursor_pos(window)
mouse_vel = np.array([0.0, 0.0], dtype='f4')


# 3. WORKER LOGIC
def worker_logic(start, end):
    p_slice = physics_data['pos'][start:end]
    v_slice = physics_data['vel'][start:end]
    pr_slice = 1.0 / props[start:end]

    while running:
        start_sim.wait()  # Wait for frame start

        # ... (Math logic remains the same) ...
        dx = target[0] - p_slice[:, 0]
        dy = target[1] - p_slice[:, 1]
        dist_sq = dx * dx + dy * dy + 60.0
        inv_dist = 1.0 / np.sqrt(dist_sq)

        # Vectorized physics update
        if is_repelling:
            force = -1200.0 * (inv_dist ** 3) * 12000.0
            ax = (dx * force + mouse_vel[0] * 8.0) * pr_slice
            ay = (dy * force + mouse_vel[1] * 8.0) * pr_slice
        else:
            force = 40.0 * inv_dist
            ax = (dx * force + mouse_vel[0] * 5.0) * pr_slice
            ay = (dy * force + mouse_vel[1] * 5.0) * pr_slice

        v_slice[:, 0] = (v_slice[:, 0] + ax * DT) * 0.99
        v_slice[:, 1] = (v_slice[:, 1] + ay * DT) * 0.99
        p_slice += v_slice

        # Screen Wrap
        p_slice[p_slice[:, 0] > 640, 0] = -640
        p_slice[p_slice[:, 0] < -640, 0] = 640
        p_slice[p_slice[:, 1] > 360, 1] = -360
        p_slice[p_slice[:, 1] < -360, 1] = 360

        done_sim.wait()  # Signal math is done


# Start Workers
chunk = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    threading.Thread(target=worker_logic, args=(i*chunk, (i+1)*chunk), daemon=True).start()

# 4. PIPELINE

vertex_shader = '''
    #version 450 core
    layout (location = 0) in vec2 in_pos;
    layout (location = 1) in vec2 in_vel;
    out float v_energy;
    void main() {
        float speed_sq = dot(in_vel, in_vel);
        v_energy = clamp(speed_sq * 0.01, 0.0, 1.0);

        // Use PointSize for visualization - much cheaper than triangles
        gl_PointSize = 1.0 + (v_energy * 2.0);
        gl_Position = vec4(in_pos / vec2(640.0, 360.0), 0.0, 1.0);
    }
'''

fragment_shader = '''
    #version 450 core
    in float v_energy;
    layout (location = 0) out vec4 out_color;
    void main() {
        // Circular point shape
        if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;

        vec3 color = mix(vec3(0.1, 0.2, 0.5), vec3(0.4, 0.9, 1.0), v_energy);
        out_color = vec4(color, 0.8);
    }
'''


pipelines = [
    ctx.pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        framebuffer=[image],
        topology='points',
        vertex_count=TRIANGLE_COUNT,
        vertex_buffers=[
            *zengl.bind(b['pos'], '2f', 0),
            *zengl.bind(b['vel'], '2f', 1),
        ],
    ) for b in buffers
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
m_prev_x, m_prev_y = glfw.get_cursor_pos(window)

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

    # --- THE PIPELINE TRICK ---
    # 1. Start the workers on the NEXT frame's data
    start_sim.set()

    # 2. While workers are "sweating", the GPU renders the CURRENT frame
    # We use (frame % 2) to pick the buffer that was finished last time
    render_idx = frame % 2
    pipelines[render_idx].render()

    # 3. Wait for workers to finish before we start the next loop
    done_sim.wait()
    start_sim.clear()

    # 4. Update the buffers for the NEXT frame
    # We write to the buffer we just finished rendering
    write_idx = render_idx
    buffers[write_idx]['pos'].write(physics_data['pos'])
    buffers[write_idx]['vel'].write(physics_data['vel'])

    # --- THE CLEANUP ---
    image.blit()
    ctx.end_frame()  # Flush to GPU

    # These MUST only happen once!
    glfw.swap_buffers(window)
    glfw.poll_events()

    frame += 1
    if frame % 60 == 0:
        t_now = time.perf_counter()
        fps = 60 * (1.0/(t_now - t_prev))
        t_prev = t_now
        glfw.set_window_title(window, f"{gil_tpl} | FPS: {int(fps)} | Particles: {TRIANGLE_COUNT}")

glfw.terminate()