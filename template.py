# TEMPLATE

import glfw
import zengl
import numpy as np
import threading
import time
import sys

# 1. Boilerplate Setup
if not glfw.init():
    sys.exit()

window = glfw.create_window(1280, 720, "ZenGL No-GIL Template", None, None)
glfw.make_context_current(window)
ctx = zengl.context()

# 2. Shared Data (The "Bridge" between Threads)
# We use a single large array for all instance data
COUNT = 100_000
# Layout: [x, y, vx, vy, energy, padding]
shared_data = np.zeros((COUNT, 6), dtype='f4')
shared_data[:, :2] = np.random.uniform(-1.0, 1.0, (COUNT, 2))  # Initial Positions


# 3. No-GIL Worker
def physics_worker():
    # Pre-slice views to avoid allocation in the loop
    pos = shared_data[:, 0:2]
    vel = shared_data[:, 2:4]

    while True:
        # Simple Physics: Move in circles
        # In No-GIL Python 3.13+, this loop runs at full speed
        # without blocking the main render thread.
        pos += vel * 0.01

        # Keep them on screen (Simple wrap)
        np.putmask(pos, pos > 1.0, -1.0)
        np.putmask(pos, pos < -1.0, 1.0)

        # High-frequency update
        time.sleep(0.001)


threading.Thread(target=physics_worker, daemon=True).start()

# 4. ZenGL Pipeline
# Minimalist shader: Just draws points at the positions
image = ctx.image((1280, 720), 'rgba8unorm')
instance_buffer = ctx.buffer(size=COUNT * 24)  # 6 floats * 4 bytes

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        layout (location = 0) in vec2 in_pos;
        void main() {
            gl_PointSize = 2.0;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core
        layout (location = 0) out vec4 out_color;
        void main() { out_color = vec4(0.2, 0.6, 1.0, 1.0); }
    ''',
    framebuffer=[image],
    topology='points',
    vertex_count=1,
    instance_count=COUNT,
    vertex_buffers=[
        *zengl.bind(instance_buffer, '2f 4x /i', 0),  # Read 2 floats, skip 4
    ],
)

# 5. Main Render Loop
while not glfw.window_should_close(window):
    ctx.new_frame()
    image.clear()

    # Fast upload of shared data to GPU
    instance_buffer.write(shared_data)
    pipeline.render()

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()