import glfw
import zengl
import numpy as np
import sys

if not glfw.init():
    sys.exit()

window = glfw.create_window(1280, 720, "3.14t | The Last Stand", None, None)
glfw.make_context_current(window)

ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

# This pipeline is STATIC. It never changes.
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        void main() {
            vec2 vertices[3] = vec2[](
                vec2(0.0, 0.5),
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5)
            );
            // We use the gl_Position directly.
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
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
)

while not glfw.window_should_close(window):
    mx, my = glfw.get_cursor_pos(window)

    # PRODUCTION TRICK:
    # Instead of moving the triangle in the shader (which is broken),
    # we move the VIEWPORT where the triangle is drawn.

    ctx.new_frame()
    image.clear()

    # We "shift" the world by changing the viewport
    # to follow the mouse.
    # Scaling pulse
    pulse = 1.0 + 0.2 * np.sin(glfw.get_time() * 5.0)
    v_w = int(1280 * pulse)
    v_h = int(720 * pulse)

    # Center it on the mouse
    pipeline.viewport = (int(mx - v_w / 2), int(720 - my - v_h / 2), v_w, v_h)
    pipeline.render()

    image.blit()
    ctx.end_frame()

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()