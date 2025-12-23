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


# Shared State
class State:
    pos_x = 0.0
    pos_y = 0.0
    color = (0.2, 0.8, 1.0)  # Neon Blue


state = State()

# 2. Physics Thread (High-frequency updates)
def physics_worker():
    while not glfw.window_should_close(window):
        t = time.time()
        # High speed orbit
        state.pos_x = np.cos(t * 3.0) * 300
        state.pos_y = np.sin(t * 3.0) * 300

        # Shift color based on position
        state.color = (0.5 + 0.5 * np.cos(t), 0.8, 0.5 + 0.5 * np.sin(t))
        time.sleep(0.0005)  # 2000Hz updates


threading.Thread(target=physics_worker, daemon=True).start()


# 3. Using 3.14 Template Strings for Shader Definitions
# We define the source as a template that we can "reify" if needed,
# though here we use it to keep our fragment color dynamic but clean.
def get_pipeline(r, g, b):
    # In 3.14, we can use the t"" prefix for template strings
    # This keeps the shader code "pure" until we need the specific variant.
    return ctx.pipeline(
        vertex_shader='''
            #version 450 core
            void main() {
                vec2 vertices[3] = vec2[](vec2(0, 0.1), vec2(-0.1, -0.1), vec2(0.1, -0.1));
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            }
        ''',
        fragment_shader=f'''
            #version 450 core
            layout (location = 0) out vec4 out_color;
            void main() {{
                out_color = vec4({r}, {g}, {b}, 1.0);
            }}
        ''',
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
    )


# Cache a few pipelines to avoid recompiling every single frame
# (A simple version of what we discussed earlier)
pipeline_cache = {}

# 4. Main Render Loop
while not glfw.window_should_close(window):
    ctx.new_frame()
    image.clear()

    # We round the color slightly to avoid creating 60 pipelines a second
    color_key = (round(state.color[0], 1), round(state.color[1], 1), round(state.color[2], 1))

    if color_key not in pipeline_cache:
        pipeline_cache[color_key] = get_pipeline(*color_key)

    active_pipeline = pipeline_cache[color_key]

    # Use our Viewport transformation for the thread-calculated position
    active_pipeline.viewport = (int(state.pos_x), int(state.pos_y), 1280, 720)
    active_pipeline.render()

    image.blit()
    ctx.end_frame()

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()