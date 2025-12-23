from typing import TYPE_CHECKING
if TYPE_CHECKING: from zengl import BlendSettings

import glfw
import zengl
import numpy as np
import sys
import threading
import time

# 1. Initialize Window & Context
if not glfw.init():
    sys.exit()

GIL_STATE: str = "No GIL" if sys._is_gil_enabled() == False else "GIL"
window = glfw.create_window(1280, 720, f"3.14t Swarm | {GIL_STATE}", None, None)
glfw.make_context_current(window)

ctx = zengl.context()
image = ctx.image((1280, 720), 'rgba8unorm')

# 2. CONFIGURATIONS
TRIANGLE_COUNT = 2000
NUM_WORKERS = 6

# Shared arrays for workers
positions = np.zeros(TRIANGLE_COUNT * 2, dtype='f4')
velocities = np.zeros(TRIANGLE_COUNT * 2, dtype='f4')
# Individual personalities: [mass, drag, offset_x, offset_y]
props = np.random.uniform(0.5, 1.5, (TRIANGLE_COUNT, 4)).astype('f4')
target = [0.0, 0.0]


# 3. WORKER LOGIC (No-GIL Parallel Physics)
is_repelling = False

def worker_logic(start_idx, end_idx):
    while not glfw.window_should_close(window):
        t = time.time()
        for i in range(start_idx, end_idx):
            # Screen Wrapping logic (World space is approx -640 to 640, -360 to 360)
            if positions[i * 2] > 700: positions[i * 2] = -700
            if positions[i * 2] < -700: positions[i * 2] = 700
            if positions[i * 2 + 1] > 400: positions[i * 2 + 1] = -400
            if positions[i * 2 + 1] < -400: positions[i * 2 + 1] = 400

            px, py = positions[i * 2], positions[i * 2 + 1]
            vx, vy = velocities[i * 2], velocities[i * 2 + 1]

            mass = props[i, 0]
            drag = 0.94 + (props[i, 1] * 0.02)
            # Each triangle has a unique "perch" around the mouse
            tx = target[0] + (props[i, 2] - 1.0) * 100.0
            ty = target[1] + (props[i, 3] - 1.0) * 100.0

            dx, dy = tx - px, ty - py
            dist = np.sqrt(dx * dx + dy * dy) + 10.0

            # Acceleration with individual mass
            # Change force direction based on global state
            force_mult = -2.0 if is_repelling else 0.8

            ax = (dx / dist) * force_mult / mass
            ay = (dy / dist) * force_mult / mass

            # Organic fluttering
            ax += np.sin(t * 10.0 + i) * 1.5  # Faster, stronger oscillation
            ay += np.cos(t * 12.0 + i) * 1.5

            velocities[i * 2] = (vx + ax) * drag
            velocities[i * 2 + 1] = (vy + ay) * drag
            positions[i * 2] += velocities[i * 2]
            positions[i * 2 + 1] += velocities[i * 2 + 1]
        time.sleep(0.005)


step = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    s, e = i * step, (TRIANGLE_COUNT if i == NUM_WORKERS - 1 else (i + 1) * step)
    threading.Thread(target=worker_logic, args=(s, e), daemon=True).start()

# 4. PIPELINE SETUP
shape_buffer = ctx.buffer(np.array([0, 0.03, -0.01, -0.01, 0.01, -0.01], dtype='f4'))
instance_buffer = ctx.buffer(size=TRIANGLE_COUNT * 16)

vertex_shader_code = '''
    #version 450 core
    layout (location = 0) in vec2 in_vert; 
    layout (location = 1) in vec4 in_inst; 
    out float v_speed;
    void main() {
        vec2 pos = in_inst.xy;
        vec2 vel = in_inst.zw;
        v_speed = length(vel);
        float angle = atan(vel.y, vel.x) - 1.5708;
        mat2 rot = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
        gl_Position = vec4((rot * in_vert) + (pos / vec2(640.0, 360.0)), 0.0, 1.0);
    }
'''

fragment_shader_code = '''
    #version 450 core
    in float v_speed;
    layout (location = 0) out vec4 out_color;
    void main() {
        vec3 low = vec3(0.1, 0.4, 0.9);
        vec3 high = vec3(1.0, 0.8, 0.3);
        out_color = vec4(mix(low, high, clamp(v_speed * 0.05, 0.0, 1.0)), 1.0);
    }
'''

pipeline = ctx.pipeline(
    vertex_shader=vertex_shader_code,
    fragment_shader=fragment_shader_code,
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
    instance_count=TRIANGLE_COUNT,
    vertex_buffers=[
        # Use * to unpack the lists returned by zengl.bind
        *zengl.bind(shape_buffer, '2f', 0),
        *zengl.bind(instance_buffer, '4f /i', 1),
    ],
)

# Fade pipeline for the Trail Effect
blend_settings: BlendSettings = {
    'enable': True,
    'src_color': 'src_alpha',
    'dst_color': 'one_minus_src_alpha',
    'src_alpha': 'one',
    'dst_alpha': 'one_minus_src_alpha',
}
fade_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 450 core
        void main() {
            vec2 v[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));
            gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 450 core
        layout (location = 0) out vec4 out_color;
        void main() { out_color = vec4(0.0, 0.0, 0.0, 0.15); } // 0.15 is trail persistence
    ''',
    framebuffer=[image],
    topology='triangle_strip',
    vertex_count=4,
    blend=blend_settings,
)

# 5. RENDER LOOP
gpu_data = np.zeros((TRIANGLE_COUNT, 4), dtype='f4')

while not glfw.window_should_close(window):
    mx, my = glfw.get_cursor_pos(window)
    is_repelling = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    target[0], target[1] = mx - 640, 360 - my

    ctx.new_frame()

    # Instead of image.clear(), we draw the fade quad
    fade_pipeline.render()

    # Update instance buffer in one big chunk
    gpu_data[:, 0:2] = positions.reshape(-1, 2)
    gpu_data[:, 2:4] = velocities.reshape(-1, 2)
    instance_buffer.write(gpu_data)

    pipeline.render()
    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()