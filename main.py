import glfw
import zengl
import numpy as np
import sys

# 1. Initialize Window
if not glfw.init():
    sys.exit()

window = glfw.create_window(1280, 720, "3.14t | ZenGL Zero-Resource", None, None)
glfw.make_context_current(window)

# 2. ZenGL Setup
ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

# We are using NO buffers. No VBO, no Uniforms.
pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core

        // We pass the time/offset through the gl_InstanceID or a constant
        // For this test, let's hardcode the vertices and use a constant.

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.866, -0.7),
            vec2(0.866, -0.7)
        );

        // We use a constant that we will replace in the string
        #define OFFSET 0.0

        void main() {
            vec2 pos = vertices[gl_VertexID];
            pos.x += OFFSET;
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
)

# 3. Main Loop
while not glfw.window_should_close(window):
    t = glfw.get_time()
    offset = np.sin(t * 3.0) * 0.4

    # Let's change the color over time too!
    r = 0.5 + 0.5 * np.sin(t)
    g = 0.5 + 0.5 * np.sin(t + 2.0)
    b = 0.5 + 0.5 * np.sin(t + 4.0)

    pipeline = ctx.pipeline(
        vertex_shader=f'''
            #version 450 core
            vec2 vertices[3] = vec2[](
                vec2(0.0 + {offset}, 0.8),
                vec2(-0.866 + {offset}, -0.7),
                vec2(0.866 + {offset}, -0.7)
            );
            void main() {{
                gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            }}
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

    ctx.new_frame()
    image.clear()
    pipeline.render()
    image.blit()
    ctx.end_frame()

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()