import glfw
import zengl
import numpy as np
import struct
import time
import json
import os
from dataclasses import dataclass

WINDOW_SIZE = (1280, 720)
MOUSE_SENSITIVITY = 0.003

cube_vertices = np.array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
    -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
    -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5,
], dtype='f4')


@dataclass
class Platform:
    x: float
    y: float
    z: float
    hw: float
    hh: float
    hd: float

    def to_dict(self): return {'x': self.x, 'y': self.y, 'z': self.z, 'hw': self.hw, 'hh': self.hh, 'hd': self.hd}


def get_perspective(fovy_deg, aspect, near, far):
    fovy = np.radians(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    res = np.zeros((4, 4), dtype='f4')
    res[0, 0] = f / aspect
    res[1, 1] = f
    res[2, 2] = (far + near) / (near - far)
    res[2, 3] = (2 * far * near) / (near - far)
    res[3, 2] = -1.0
    return res


def get_lookat(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    res = np.eye(4, dtype='f4')
    res[0, :3] = x
    res[1, :3] = y
    res[2, :3] = z
    res[0, 3] = -np.dot(x, eye)
    res[1, 3] = -np.dot(y, eye)
    res[2, 3] = -np.dot(z, eye)
    return res


def get_model_matrix(x, y, z, sx, sy, sz):
    m = np.eye(4, dtype='f4');
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz;
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def ray_aabb_intersect(origin, direction, p_min, p_max):
    t1 = (p_min - origin) / (direction + 1e-6);
    t2 = (p_max - origin) / (direction + 1e-6)
    t_min = np.maximum(np.minimum(t1, t2), 0.0);
    t_max = np.minimum(np.maximum(t1, t2), 1000.0)
    t_near = np.max(t_min);
    t_far = np.min(t_max)
    if t_far < t_near: return False, 0.0
    return True, t_near


class Editor:
    def __init__(self, platforms):
        self.platforms = platforms
        self.selected_index = -1
        self.cam_x, self.cam_y, self.cam_z = 0.0, 5.0, 10.0
        self.cam_yaw, self.cam_pitch = -1.57, -0.5

    def update(self, window, dt):
        speed = 10.0 * dt
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS: speed = 30.0 * dt
        cx, cz = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        rx, rz = np.cos(self.cam_yaw + np.pi / 2), np.sin(self.cam_yaw + np.pi / 2)

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: self.cam_x += cx * speed; self.cam_z += cz * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: self.cam_x -= cx * speed; self.cam_z -= cz * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: self.cam_x -= rx * speed; self.cam_z -= rz * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: self.cam_x += rx * speed; self.cam_z += rz * speed
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS: self.cam_y += speed
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS: self.cam_y -= speed

    def select(self):
        dx = np.cos(self.cam_yaw) * np.cos(self.cam_pitch)
        dy = np.sin(self.cam_pitch)
        dz = np.sin(self.cam_yaw) * np.cos(self.cam_pitch)
        origin = np.array([self.cam_x, self.cam_y, self.cam_z])
        direction = np.array([dx, dy, dz])

        closest, idx = 999.0, -1
        for i, p in enumerate(self.platforms):
            p_min = np.array([p.x - p.hw, p.y - p.hh, p.z - p.hd])
            p_max = np.array([p.x + p.hw, p.y + p.hh, p.z + p.hd])
            hit, t = ray_aabb_intersect(origin, direction, p_min, p_max)
            if hit and t < closest: closest, idx = t, i
        self.selected_index = idx
        print(f"Selected: {idx}")

    def on_key(self, window, key, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_N:
                dist = 5.0
                nx = self.cam_x + np.cos(self.cam_yaw) * dist
                ny = self.cam_y + np.sin(self.cam_pitch) * dist
                nz = self.cam_z + np.sin(self.cam_yaw) * dist
                nx, ny, nz = round(nx * 2) / 2, round(ny * 2) / 2, round(nz * 2) / 2
                self.platforms.append(Platform(nx, ny, nz, 1.0, 0.2, 1.0))
                self.selected_index = len(self.platforms) - 1
            if key == glfw.KEY_X and self.selected_index != -1:
                self.platforms.pop(self.selected_index)
                self.selected_index = -1
            if key == glfw.KEY_S:
                with open('level.json', 'w') as f: json.dump([p.to_dict() for p in self.platforms], f)
                print("Saved level.json")

        if action in [glfw.PRESS, glfw.REPEAT] and self.selected_index != -1:
            p = self.platforms[self.selected_index]
            step = 0.5
            is_resize = (mods & glfw.MOD_SHIFT)
            if key == glfw.KEY_UP:
                if is_resize:
                    p.hd += 0.1
                else:
                    p.z -= step
            elif key == glfw.KEY_DOWN:
                if is_resize:
                    p.hd = max(0.1, p.hd - 0.1)
                else:
                    p.z += step
            elif key == glfw.KEY_LEFT:
                if is_resize:
                    p.hw = max(0.1, p.hw - 0.1)
                else:
                    p.x -= step
            elif key == glfw.KEY_RIGHT:
                if is_resize:
                    p.hw += 0.1
                else:
                    p.x += step
            elif key == glfw.KEY_PAGE_UP:
                if is_resize:
                    p.hh += 0.1
                else:
                    p.y += step
            elif key == glfw.KEY_PAGE_DOWN:
                if is_resize:
                    p.hh = max(0.05, p.hh - 0.1)
                else:
                    p.y -= step


if __name__ == "__main__":
    glfw.init()
    window = glfw.create_window(1280, 720, "Level Editor", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)

    platforms = []
    if os.path.exists("level.json"):
        with open("level.json", "r") as f:
            for d in json.load(f): platforms.append(Platform(**d))
    else:
        platforms = [Platform(0, -1, 0, 4, 0.2, 4)]

    editor = Editor(platforms)
    ctx = zengl.context()
    image = ctx.image(WINDOW_SIZE, 'rgba8unorm')
    depth = ctx.image(WINDOW_SIZE, 'depth24plus')
    vbo = ctx.buffer(cube_vertices)
    pipeline = ctx.pipeline(
        vertex_shader='''
            #version 330 core
            layout (location = 0) in vec3 in_vert;
            uniform mat4 mvp; uniform mat4 model; uniform vec3 color;
            out vec3 v_color; out vec3 v_world_pos;
            void main() { v_color = color; v_world_pos = vec3(model * vec4(in_vert, 1.0)); gl_Position = mvp * vec4(in_vert, 1.0); }
        ''',
        fragment_shader='''
            #version 330 core
            out vec4 out_color; in vec3 v_color; in vec3 v_world_pos; uniform int is_selected;
            void main() {
                vec3 dx = dFdx(v_world_pos); vec3 dy = dFdy(v_world_pos); vec3 normal = normalize(cross(dx, dy));
                float diff = max(dot(normal, normalize(vec3(0.5, 0.8, 0.3))), 0.0);
                vec3 final = v_color * (diff + 0.5) + (v_color * vec3(0.2));
                if (is_selected == 1) final += vec3(0.3, 0.3, 0.0);
                out_color = vec4(final, 1.0);
            }
        ''',
        framebuffer=[image, depth], topology='triangles', cull_face='none', depth={'func': 'less', 'write': True},
        vertex_buffers=[*zengl.bind(vbo, '3f', 0)],
        uniforms={'mvp': [0.0] * 16, 'model': [0.0] * 16, 'color': [1.0, 1.0, 1.0], 'is_selected': 0}, vertex_count=36
    )


    def on_key(win, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)
        editor.on_key(win, key, action, mods)


    def on_mouse_click(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS: editor.select()


    def on_mouse_move(win, x, y):
        if not hasattr(on_mouse_move, 'lx'): on_mouse_move.lx, on_mouse_move.ly = x, y
        dx, dy = x - on_mouse_move.lx, y - on_mouse_move.ly
        on_mouse_move.lx, on_mouse_move.ly = x, y
        editor.cam_yaw += dx * 0.003
        editor.cam_pitch += dy * 0.003
        editor.cam_pitch = max(-1.5, min(1.5, editor.cam_pitch))


    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_mouse_button_callback(window, on_mouse_click)

    last_time = time.perf_counter()
    while not glfw.window_should_close(window):
        t = time.perf_counter()
        dt = t - last_time
        last_time = t
        if dt > 0.1: dt = 0.1

        editor.update(window, dt)

        image.clear_value = (0.1, 0.1, 0.15, 1.0)
        image.clear()
        depth.clear()
        aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
        proj = get_perspective(60.0, aspect, 0.1, 100.0)

        cx, cy, cz = editor.cam_x, editor.cam_y, editor.cam_z
        tx = cx + np.cos(editor.cam_yaw) * np.cos(editor.cam_pitch)
        ty = cy + np.sin(editor.cam_pitch)
        tz = cz + np.sin(editor.cam_yaw) * np.cos(editor.cam_pitch)
        view = get_lookat(np.array([cx, cy, cz], dtype='f4'), np.array([tx, ty, tz], dtype='f4'),
                          np.array([0, 1, 0], dtype='f4'))

        for i, p in enumerate(editor.platforms):
            m = get_model_matrix(p.x, p.y, p.z, p.hw * 2, p.hh * 2, p.hd * 2)
            col = (0.3, 0.7, 0.4)
            pipeline.uniforms['mvp'][:] = (proj @ view @ m).T.tobytes()
            pipeline.uniforms['model'][:] = m.T.tobytes()
            pipeline.uniforms['color'][:] = struct.pack('3f', *col)
            pipeline.uniforms['is_selected'][:] = struct.pack('i', 1 if i == editor.selected_index else 0)
            pipeline.render()

        image.blit()
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()