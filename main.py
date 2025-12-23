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
gil_tpl = f"3.14t Swarm | {GIL_STATE}"
window = glfw.create_window(1280, 720, gil_tpl, None, None)
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
target = np.array([0.0, 0.0], dtype='f4') # micro-optimization


# 3. WORKER LOGIC (No-GIL Parallel Physics)
is_repelling = False

# Updated Shared Array: [x, y, vx, vy] repeated for TRIANGLE_COUNT
gpu_data = np.zeros((TRIANGLE_COUNT, 6), dtype='f4')


def worker_logic(start_idx, end_idx):
    last_time = time.perf_counter()

    # Force a direct reference to the shared memory
    my_data = gpu_data[start_idx:end_idx]
    my_props = props[start_idx:end_idx]

    while running:
        current_time = time.perf_counter()
        dt = min(current_time - last_time, 0.05)
        last_time = current_time
        speed_scale = dt * 60.0

        for i in range(len(my_data)):
            px, py = my_data[i, 0], my_data[i, 1]
            vx, vy = my_data[i, 2], my_data[i, 3]

            # data structure: [0]=pos.x, [1]=pos.y, [2]=vel.x, [3]=vel.y
            dx = target[0] - my_data[i, 0]
            dy = target[1] - my_data[i, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1.0

            mass = my_props[i, 0]  # Now used!
            drag = 0.94 + (my_props[i, 1] * 0.02)

            if is_repelling:
                mag = -15.0 / (dist / 50.0 + 1.0)
            else:
                mag = 0.5 / (dist / 100.0 + 1.0)

            # Divide force by mass (f=ma -> a=f/m)
            ax = ((dx / dist) * mag) / mass
            ay = ((dy / dist) * mag) / mass

            if not is_repelling:
                swirl = 2.0 / (dist / 100.0 + 1.0)
                ax += ((-dy / dist) * swirl) / mass
                ay += ((dx / dist) * swirl) / mass

            new_vx = (vx + ax * speed_scale) * (drag ** speed_scale)
            new_vy = (vy + ay * speed_scale) * (drag ** speed_scale)

            my_data[i, 2] = new_vx
            my_data[i, 3] = new_vy
            my_data[i, 0] = px + new_vx * speed_scale
            my_data[i, 1] = py + new_vy * speed_scale

            # 4. ENERGY calculation
            dist = np.sqrt(dx * dx + dy * dy) + 1.0
            my_data[i, 4] = 0.5 * mass * (new_vx ** 2 + new_vy ** 2)  # Kinetic
            my_data[i, 5] = mass * (dist * 0.01)  # Potential

            # Boundary Wrap
            if my_data[i, 0] > 650:
                my_data[i, 0] = -650
            elif my_data[i, 0] < -640:
                my_data[i, 0] = 650
            if my_data[i, 1] > 370:
                my_data[i, 1] = -370
            elif my_data[i, 1] < -370:
                my_data[i, 1] = 370

        time.sleep(0.001)

# Random positions between -600 and 600
gpu_data[:, 0] = np.random.uniform(-600, 600, TRIANGLE_COUNT)
gpu_data[:, 1] = np.random.uniform(-350, 350, TRIANGLE_COUNT)
# Small random initial velocity so they aren't "static"
gpu_data[:, 2:4] = np.random.uniform(-1, 1, (TRIANGLE_COUNT, 2))

# Use a flag instead of glfw call in workers
running = True

step = TRIANGLE_COUNT // NUM_WORKERS
for i in range(NUM_WORKERS):
    s, e = i * step, (TRIANGLE_COUNT if i == NUM_WORKERS - 1 else (i + 1) * step)
    threading.Thread(target=worker_logic, args=(s, e), daemon=True).start()

# 4. PIPELINE SETUP
shape_buffer = ctx.buffer(np.array([0, 0.03, -0.01, -0.01, 0.01, -0.01], dtype='f4'))
instance_buffer = ctx.buffer(size=TRIANGLE_COUNT * 24)

vertex_shader_code = '''
    #version 450 core
    layout (location = 0) in vec2 in_vert;   // The triangle mesh
    layout (location = 1) in vec4 in_inst;   // x, y, vx, vy
    layout (location = 2) in vec2 in_energy; // kinetic, potential
    
    out float v_energy; // Sending this to the fragment shader
    
    void main() {
        // 1. Assign values from our input vectors
        vec2 pos = in_inst.xy;
        vec2 vel = in_inst.zw;
        
        // 2. Calculate Total Energy for the fragment shader
        // We can also normalize it here so it's 0.0 to 1.0
        // Adjust 0.005 based on how "bright" you want the swarm to be
        v_energy = clamp((in_energy.x + in_energy.y) * 0.005, 0.0, 1.0);
    
        // 3. Rotation logic based on velocity
        float angle = atan(vel.y, vel.x) - 1.5708;
        mat2 rot = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
        
        // 4. Final Position
        gl_Position = vec4((rot * in_vert) + (pos / vec2(640.0, 360.0)), 0.0, 1.0);
    }
'''

fragment_shader_code = '''
    #version 450 core
    in float v_energy;
    layout (location = 0) out vec4 out_color;
    
    void main() {
        // Blue for low energy, Red/Orange for high kinetic/potential energy
        vec3 cold = vec3(0.05, 0.1, 0.4); 
        vec3 hot = vec3(1.0, 0.9, 0.4);
        
        out_color = vec4(mix(cold, hot, v_energy), 1.0);
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
        *zengl.bind(shape_buffer, '2f', 0),
        # '4f 2f /i' means:
        # Read 4 floats for location 1, then 2 floats for location 2, then move to next instance
        *zengl.bind(instance_buffer, '4f 2f /i', 1, 2),
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

# Create two buffers and two pipelines tied to them
instance_buffers = [
    ctx.buffer(size=TRIANGLE_COUNT * 24),
    ctx.buffer(size=TRIANGLE_COUNT * 24)
]

swarm_pipelines = [
    ctx.pipeline(
        vertex_shader=vertex_shader_code,
        fragment_shader=fragment_shader_code,
        framebuffer=[image],
        topology='triangles',
        vertex_count=3,
        instance_count=TRIANGLE_COUNT,
        vertex_buffers=[
            *zengl.bind(shape_buffer, '2f', 0),
            # MAKE SURE THIS SAYS instance_buffers[i] (with an 's')
            *zengl.bind(instance_buffers[i], '4f 2f /i', 1, 2),
        ],
    ) for i in range(2)
]

# 5. RENDER LOOP

prev_frame_time = 0

frame_idx = 0
while not glfw.window_should_close(window):
    mx, my = glfw.get_cursor_pos(window)
    is_repelling = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    target[0], target[1] = mx - 640, 360 - my

    ctx.new_frame()
    fade_pipeline.render()

    # --- PING-PONG BUFFERS ---
    curr_idx = frame_idx % 2

    # 1. Write the pre-formatted data to the next buffer
    instance_buffers[curr_idx].write(gpu_data)

    # 2. Render from that buffer
    swarm_pipelines[curr_idx].render()

    # 3. Force Sync (The "Truth" check)
    _ = image.read(size=(1, 1))

    # --- METRICS ---
    now = time.perf_counter()
    fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 0
    prev_frame_time = now
    glfw.set_window_title(window, f"{gil_tpl} | True FPS: {int(fps)}")

    image.blit()
    ctx.end_frame()
    glfw.swap_buffers(window)
    glfw.poll_events()
    frame_idx += 1

glfw.terminate()