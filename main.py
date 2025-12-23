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
TRIANGLE_COUNT = 50
NUM_WORKERS = 8        # Match this to your CPU cores

# A shared NumPy array for all positions: [x, y, x, y, ...]
# NumPy arrays are great for 3.14t because they are essentially "raw" memory
positions = np.zeros(TRIANGLE_COUNT * 2, dtype='f4')
# --- Add a Velocity Array ---
# Same size as positions, stores [vx, vy, vx, vy...]
velocities = np.zeros(TRIANGLE_COUNT * 2, dtype='f4')

# Shared target: [x, y]
target = [0.0, 0.0]

# 3. WORKER LOGIC
# --- Better Chunking ---
# We calculate the chunks but ensure the last worker takes the "remainder"
chunks = []
step = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    start = i * step
    # If it's the last worker, go all the way to TRIANGLE_COUNT
    end = TRIANGLE_COUNT if i == NUM_WORKERS - 1 else (i + 1) * step
    chunks.append((start, end))




def worker_logic(start_idx, end_idx):
    while not glfw.window_should_close(window):
        t = time.time()
        for i in range(start_idx, end_idx):
            # 1. Get Current Data
            px, py = positions[i * 2], positions[i * 2 + 1]
            vx, vy = velocities[i * 2], velocities[i * 2 + 1]

            # 2. Calculate Vector to Mouse (The "Pull")
            dx = target[0] - px
            dy = target[1] - py
            dist = np.sqrt(dx * dx + dy * dy) + 0.1  # avoid div by zero

            # Normalize and apply "Gravity" force (this section might be important mathematically, but I'm stupid
            force = 0.5
            ax = (dx / dist) * force
            ay = (dy / dist) * force

            # 3. Add "Turbulence" (The "Organic" part)
            # This replaces the fan math with random-ish fluttering
            ax += np.sin(t * 2.0 + i) * 0.2
            ay += np.cos(t * 3.0 + i) * 0.2

            # 4. Update Velocity (Acceleration -> Velocity)
            vx += ax
            vy += ay

            # 5. Apply Drag/Friction (Prevents them from flying off to infinity)
            vx *= 0.95
            vy *= 0.95

            # 6. Update Position (Velocity -> Position)
            positions[i * 2] += vx
            positions[i * 2 + 1] += vy

        time.sleep(0.001)


# Spawn workers using our safe chunks
for s, e in chunks:
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
    mx, my = glfw.get_cursor_pos(window)
    # Convert window pixels to our coordinate system
    target[0] = mx - 640
    target[1] = 360 - my

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