import glfw
import zengl
import numpy as np
import sys
import threading
import time

# 1. Initialize Window & Context
if not glfw.init():
    sys.exit()

GIL_STATE: str = "No GIL" if sys._is_gil_enabled()==False else "GIL"
gil_tpl: str = f"3.14t | Template Strings + {GIL_STATE}"

window = glfw.create_window(1280, 720, gil_tpl, None, None)
glfw.make_context_current(window)

ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

# 2. CONFIGURATIONS
TRIANGLE_COUNT = 2000
NUM_WORKERS = 8        # Match this to your CPU cores

# A shared NumPy array for all positions: [x, y, x, y, ...]
# NumPy arrays are great for 3.14t because they are essentially "raw" memory
positions = np.zeros(TRIANGLE_COUNT * 2, dtype='f4')

# 3. WORKER LOGIC (Place this before the render loop)
def worker_logic(start_idx, end_idx):
    """Handles a slice of the triangle array."""
    while not glfw.window_should_close(window):
        t = time.time()
        for i in range(start_idx, end_idx):
            # Calculate unique movement for this specific triangle
            off = i * 0.05
            radius = 50 + (i * 0.15)
            # Update the shared array directly
            positions[i*2] = np.cos(t + off) * radius
            positions[i*2 + 1] = np.sin(t + off) * radius
        # Tiny sleep to prevent 100% CPU pinning if desired
        time.sleep(0.001)

# Spawn the workers
chunk_size = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    s, e = i * chunk_size, (i + 1) * chunk_size
    threading.Thread(target=worker_logic, args=(s, e), daemon=True).start()

# 4. RE-USE THE SAME PIPELINE (Performance trick)
# We use one pipeline and just change the viewport per draw call
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        void main() {
            vec2 v[3] = vec2[](vec2(0, 0.02), vec2(-0.015, -0.015), vec2(0.015, -0.015));
            gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core
        layout (location = 0) out vec4 out_color;
        void main() { out_color = vec4(0.3, 0.7, 1.0, 1.0); }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

# 5. RENDER LOOP
while not glfw.window_should_close(window):
    ctx.new_frame()
    image.clear()

    # Draw all triangles using the data filled by workers
    for i in range(TRIANGLE_COUNT):
        tx = positions[i * 2]
        ty = positions[i * 2 + 1]

        # Use the viewport trick to place the triangle
        pipeline.viewport = (int(tx), int(ty), 1280, 720)
        pipeline.render()

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()