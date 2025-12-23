import glfw
import zengl
import numpy as np
import sys
import threading
import time
import random

# 1. Initialize Window & Context
if not glfw.init():
    sys.exit()

GIL_STATE: str = "No GIL" if sys._is_gil_enabled()==False else "GIL"
gil_tpl: str = f"3.14t | Template Strings + {GIL_STATE}"

window = glfw.create_window(1280, 720, gil_tpl, None, None)
glfw.make_context_current(window)

ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

TRIANGLE_COUNT = 50
triangles = []


def triangle_logic(index):
    """Each triangle has its own thread and its own logic loop."""
    angle = random.random() * np.pi * 2
    speed = 1.0 + random.random() * 2.0
    radius = 100 + random.random() * 200

    while not glfw.window_should_close(window):
        t = time.time() * speed
        # Update this triangle's specific state
        triangles[index]['x'] = np.cos(t + angle) * radius
        triangles[index]['y'] = np.sin(t + angle) * radius
        # Just a tiny sleep to yield
        time.sleep(0.001)


# Initialize states and start threads
for i in range(TRIANGLE_COUNT):
    triangles.append({'x': 0.0, 'y': 0.0, 'color': (random.random(), random.random(), 1.0)})
    threading.Thread(target=triangle_logic, args=(i,), daemon=True).start()


# 3. Pipeline Generator (Still avoiding the broken uniform validator)
def create_simple_pipeline(r, g, b):
    return ctx.pipeline(
        vertex_shader='''
            #version 450 core
            void main() {
                vec2 v[3] = vec2[](vec2(0, 0.05), vec2(-0.04, -0.04), vec2(0.04, -0.04));
                gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader=f'''
            #version 450 core
            layout (location = 0) out vec4 out_color;
            void main() {{ out_color = vec4({r:.2f}, {g:.2f}, {b:.2f}, 1.0); }}
        ''',
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )


# Pre-cache a few colored pipelines
pipeline_pool = [create_simple_pipeline(*t['color']) for t in triangles]

# 4. Render Loop
while not glfw.window_should_close(window):
    ctx.new_frame()
    image.clear()

    # Draw all 50 triangles using their thread-updated positions
    for i in range(TRIANGLE_COUNT):
        p = pipeline_pool[i]
        t_data = triangles[i]
        # Viewport trick to move each one independently
        p.viewport = (int(t_data['x']), int(t_data['y']), 1280, 720)
        p.render()

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()