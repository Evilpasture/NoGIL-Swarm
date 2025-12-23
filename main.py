import glfw
import zengl
import numpy as np
import sys

# 1. Initialize the Window
if not glfw.init():
    sys.exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(1280, 720, "3.14t | ZenGL + GLFW", None, None)
if not window:
    glfw.terminate()
    sys.exit()

glfw.make_context_current(window)

# 2. ZenGL Setup
ctx = zengl.context()

# Create a custom image to render into (instead of ctx.screen)
image = ctx.image((1280, 720), 'rgba8unorm')

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        // We pass the wobble value directly as a uniform that doesn't need a block
        uniform float wobble; 

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.866, -0.7),
            vec2(0.866, -0.7)
        );

        void main() {
            vec2 pos = vertices[gl_VertexID];
            pos.x += wobble; // Use the direct uniform
            gl_Position = vec4(pos, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core
        layout (location = 0) out vec4 out_color;
        void main() {
            out_color = vec4(1.0, 0.5, 0.2, 1.0);
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
    # We do NOT define resources here to avoid the validation bug
)

# 3. Main Loop
print(f"Engine Started. GIL: {not sys._is_gil_enabled()}")

while not glfw.window_should_close(window):
    # Calculate wobble value as a 4-byte float (f4)
    wobble_val = np.sin(glfw.get_time() * 3.0) * 0.4
    wobble_bytes = np.array([wobble_val], dtype='f4').tobytes()

    ctx.new_frame()
    image.clear()

    # Pass the bytes as a POSITIONAL argument (no keyword)
    pipeline.render(wobble_bytes)

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()