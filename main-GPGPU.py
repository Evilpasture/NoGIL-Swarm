import sys
import time
import struct
import glfw
import zengl
import numpy as np

# --- CONFIGURATION ---
TRIANGLE_COUNT = 50000000
width_tex = 16384
height_tex = (TRIANGLE_COUNT + width_tex - 1) // width_tex

# 1. Initialize Window & Context
if not glfw.init():
    sys.exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.RESIZABLE, False)

# Disable VSync for benchmarking
glfw.window_hint(glfw.DOUBLEBUFFER, True)
window = glfw.create_window(1280, 720, "ZenGL GPGPU Particles", None, None)
glfw.make_context_current(window)
glfw.swap_interval(0)

ctx = zengl.context()

# 2. RESOURCES
# Screen output
image = ctx.image((1280, 720), 'rgba8unorm')

# --- GPGPU SETUP ---
# We store Pos X, Pos Y, Vel X, Vel Y in one RGBA32F texture.
# We use two textures to ping-pong (Read from A -> Write to B, then swap).
texture_size = (width_tex, height_tex)

# Initial State Generation
# Pos: Random (-600, 600), Vel: Random (-1, 1)
pos_data = np.random.uniform(-600, 600, (TRIANGLE_COUNT, 2)).astype('f4')
vel_data = np.random.uniform(-1, 1, (TRIANGLE_COUNT, 2)).astype('f4')
# Properties: Inverse Mass (0.8 to 1.2)
inv_props = (1.0 / np.random.uniform(0.8, 1.2, TRIANGLE_COUNT)).astype('f4')

# Pack Initial Data: R=Px, G=Py, B=Vx, A=Vy
# Resize arrays to fit the texture padding
full_size = width_tex * height_tex
padded_pos = np.zeros((full_size, 2), dtype='f4')
padded_vel = np.zeros((full_size, 2), dtype='f4')
padded_props = np.zeros((full_size, 1), dtype='f4')

padded_pos[:TRIANGLE_COUNT] = pos_data
padded_vel[:TRIANGLE_COUNT] = vel_data
padded_props[:TRIANGLE_COUNT] = inv_props.reshape(-1, 1)

# Merge into RGBA structure
initial_state = np.dstack((padded_pos, padded_vel)).flatten().tobytes()
prop_bytes = padded_props.flatten().tobytes()

# State Textures (Ping-Pong)
tex_a = ctx.image(texture_size, 'rgba32float', data=initial_state)
tex_b = ctx.image(texture_size, 'rgba32float', data=initial_state)

# Static Texture for Mass/Props (Red channel only)
tex_props = ctx.image(texture_size, 'r32float', data=prop_bytes)

# 3. PIPELINES

# --- PHYSICS PIPELINE ---
# This renders a full-screen quad over the TEXTURE, not the screen.
# It reads the previous state and writes the new state.
physics_shader = '''
    #version 450 core

    uniform sampler2D State;
    uniform sampler2D Props;

    uniform vec2 Target;
    uniform vec2 MouseVel;
    uniform bool Repel;
    uniform float Dt;

    layout (location = 0) out vec4 out_state;

    void main() {
        // Current pixel coordinates (corresponds to particle ID)
        ivec2 coord = ivec2(gl_FragCoord.xy);

        // Read previous data
        vec4 data = texelFetch(State, coord, 0);
        float ip = texelFetch(Props, coord, 0).r;

        vec2 p = data.xy;
        vec2 v = data.zw;

        // Physics Logic (Ported exactly from your Python script)
        vec2 d = Target - p;
        float dist_sq = dot(d, d) + 60.0;
        float inv_dist = 1.0 / sqrt(dist_sq);

        float force;
        if (Repel) {
            force = -1200.0 * (inv_dist * inv_dist * inv_dist) * 12000.0;
        } else {
            force = 40.0 * inv_dist;
        }

        vec2 acc = (d * force + MouseVel * 5.0) * ip;

        v = (v + acc * Dt) * 0.99;
        p = p + v;

        // Boundary Wrap
        // (p + 640) % 1280 - 640  ->  mod(p + 640, 1280) - 640
        p = mod(p + vec2(640.0, 360.0), vec2(1280.0, 720.0)) - vec2(640.0, 360.0);

        out_state = vec4(p, v);
    }
'''

# Standard fullscreen quad vertex shader
quad_vs = '''
    #version 450 core
    vec2 positions[4] = vec2[](
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0,  1.0), vec2(1.0,  1.0)
    );
    void main() {
        gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
    }
'''

step_pipelines = [
    ctx.pipeline(
        vertex_shader=quad_vs,
        fragment_shader=physics_shader,
        layout=[
            {'name': 'State', 'binding': 0},
            {'name': 'Props', 'binding': 1},
        ],
        resources=[
            {'type': 'sampler', 'binding': 0, 'image': tex_a},
            {'type': 'sampler', 'binding': 1, 'image': tex_props},
        ],
        uniforms={
            'Dt': 0.016,
            'Target': [0.0, 0.0],
            'MouseVel': [0.0, 0.0],
            'Repel': False,
        },
        framebuffer=[tex_b],  # Write to B
        topology='triangle_strip',
        vertex_count=4,
    ),
    ctx.pipeline(
        vertex_shader=quad_vs,
        fragment_shader=physics_shader,
        layout=[
            {'name': 'State', 'binding': 0},
            {'name': 'Props', 'binding': 1},
        ],
        resources=[
            {'type': 'sampler', 'binding': 0, 'image': tex_b},  # Read from B
            {'type': 'sampler', 'binding': 1, 'image': tex_props},
        ],
        uniforms={
            'Dt': 0.016,
            'Target': [0.0, 0.0],
            'MouseVel': [0.0, 0.0],
            'Repel': False,
        },
        framebuffer=[tex_a],  # Write to A
        topology='triangle_strip',
        vertex_count=4,
    )
]

# --- RENDER PIPELINE ---
# Reads positions AND velocity from the texture to draw points
render_vs = f'''
    #version 450 core

    uniform sampler2D State;
    uniform int WidthMask; // 16383
    uniform int WidthShift; // 14

    // Send color to fragment shader
    out vec3 v_Color;

    void main() {{
        int id = gl_VertexID;
        ivec2 coord = ivec2(id & WidthMask, id >> WidthShift);

        vec4 data = texelFetch(State, coord, 0);
        vec2 pos = data.xy;
        vec2 vel = data.zw;

        // Calculate speed for coloring
        float speed = length(vel);

        // Mix between Blue (slow) and Red/White (fast)
        // Adjust the "0.05" divisor to tune sensitivity
        float t = clamp(speed * 0.05, 0.0, 1.0);
        v_Color = mix(vec3(0.1, 0.4, 1.0), vec3(1.0, 0.8, 0.6), t);

        gl_PointSize = 1.0;
        gl_Position = vec4(pos / vec2(640.0, 360.0), 0.0, 1.0);
    }}
'''

render_fs = '''
    #version 450 core

    in vec3 v_Color;
    layout (location = 0) out vec4 out_color;

    void main() {
        // Use the calculated color with some transparency
        out_color = vec4(v_Color, 0.8);
    }
'''

render_pipelines = [
    ctx.pipeline(
        vertex_shader=render_vs,
        fragment_shader=render_fs,
        layout=[{'name': 'State', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': tex_b}],  # Render state B (most recent)
        uniforms={
            'WidthMask': 16383,
            'WidthShift': 14
        },
        framebuffer=[image],
        topology='points',
        vertex_count=TRIANGLE_COUNT,
        blend={
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha'
        },
    ),
    ctx.pipeline(
        vertex_shader=render_vs,
        fragment_shader=render_fs,
        layout=[{'name': 'State', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': tex_a}],  # Render state A
        uniforms={
            'WidthMask': 16383,
            'WidthShift': 14
        },
        framebuffer=[image],
        topology='points',
        vertex_count=TRIANGLE_COUNT,
        blend={
            'enable': True,
            'src_color': 'src_alpha',
            'dst_color': 'one_minus_src_alpha'
        },
    )
]

# Trail effect (Fade out)
fade_pipe = ctx.pipeline(
    vertex_shader=quad_vs,
    fragment_shader='''
        #version 450 core
        out vec4 c;
        void main() { c = vec4(0, 0, 0, 0.15); }
    ''',
    framebuffer=[image],
    topology='triangle_strip',
    vertex_count=4,
    blend={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha'
    },
)

# 4. LOOP
frame = 0
t_prev = time.perf_counter()
m_prev_x, m_prev_y = glfw.get_cursor_pos(window)

while not glfw.window_should_close(window):
    # Input Handling
    mx, my = glfw.get_cursor_pos(window)
    m_vel_x = mx - m_prev_x
    m_vel_y = (360 - my) - (360 - m_prev_y)
    m_prev_x, m_prev_y = mx, my

    is_repelling = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    target_x = mx - 640
    target_y = 360 - my

    # Update Output Buffer (Screen)
    ctx.new_frame()
    fade_pipe.render()

    # Determine Ping-Pong Order
    idx = frame % 2

    # 1. Run Physics (GPU)
    # Update uniforms using struct.pack to convert Python types to raw bytes
    # '2f' = 2 floats (vec2), 'i' = 1 integer (bool is stored as int)
    step_pipelines[idx].uniforms['Target'][:] = struct.pack('2f', target_x, target_y)
    step_pipelines[idx].uniforms['MouseVel'][:] = struct.pack('2f', m_vel_x, m_vel_y)
    step_pipelines[idx].uniforms['Repel'][:] = struct.pack('i', int(is_repelling))

    step_pipelines[idx].render()

    # 2. Render Points (GPU)
    render_pipelines[idx].render()

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

    frame += 1
    if frame % 60 == 0:
        t_now = time.perf_counter()
        fps = 60 / (t_now - t_prev)
        t_prev = t_now
        glfw.set_window_title(window, f"ZenGL GPGPU | FPS: {int(fps)} | Particles: {TRIANGLE_COUNT}")

glfw.terminate()