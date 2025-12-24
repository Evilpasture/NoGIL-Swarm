import glfw
import zengl
import numpy as np
import threading
import time
import struct
import sys

from dataclasses import dataclass
from collections import defaultdict

# verify
is_free_threaded = hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() == False
GIL_STATE = "No GIL (True Parallelism)" if is_free_threaded else "GIL Active"
print(f"Python {sys.version.split()[0]} | {GIL_STATE}")

@dataclass
class Platform:
    x: float
    y: float
    hw: float
    hh: float


class PhysicsEngine:
    """This gives objects the physical properties they should have"""
    def __init__(self, platforms):
        self.player_data = np.array([0.0, 0.0, 0.0, 0.0], dtype='f4')
        self.prev_player_pos = np.array([0.0, 0.0], dtype='f4')
        self.platforms = platforms
        self.pw, self.ph = 0.05, 0.05
        self.keys = {}
        self.running = True

        # SPATIAL GRID SETUP
        self.cell_size = 0.5
        self.grid = defaultdict(list)
        self._build_grid()

        self.alpha = 0.0

    def _build_grid(self):
        for p in self.platforms:
            min_cx = int((p.x - p.hw) // self.cell_size)
            max_cx = int((p.x + p.hw) // self.cell_size)
            min_cy = int((p.y - p.hh) // self.cell_size)
            max_cy = int((p.y + p.hh) // self.cell_size)

            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    self.grid[(cx, cy)].append(p)

    def step_physics(self, dt):
        """Now called with a FIXED dt (e.g., 0.01) every time"""
        # Save current position as previous before moving
        self.prev_player_pos[:] = self.player_data[:2]

        px, py, vx, vy = self.player_data

        gravity = -0.22
        # Friction is now constant because dt is constant!
        friction = 0.85

        vy += gravity * dt
        vx *= friction

        if self.keys.get(glfw.KEY_LEFT): vx -= 0.0025
        if self.keys.get(glfw.KEY_RIGHT): vx += 0.0025

        new_x, new_y = px + vx, py + vy
        on_ground = False

        cx, cy = int(new_x // self.cell_size), int(new_y // self.cell_size)

        # Use a simple list for speed; the duplicate checks in collision are cheap
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat in self.grid.get((cx + dx, cy + dy), []):
                    rx, ry, rw, rh = plat.x, plat.y, plat.hw, plat.hh

                    if (abs(new_x - rx) < (rw + self.pw)) and (abs(new_y - ry) < (rh + self.ph)):
                        if abs(px - rx) < (rw + self.pw - 0.02):
                            if vy < 0 and py >= (ry + rh - 0.02):
                                new_y, vy, on_ground = ry + rh + self.ph, 0, True
                            elif vy > 0 and py <= (ry - rh + 0.02):
                                new_y, vy = ry - rh - self.ph, 0
                        elif abs(py - ry) < (rh + self.ph - 0.02):
                            new_x, vx = (rx - rw - self.pw if vx > 0 else rx + rw + self.pw), 0

        if self.keys.get(glfw.KEY_UP) and on_ground:
            vy = 0.045

        self.player_data[:] = [new_x, new_y, vx, vy]

    def update(self):
        """The Accumulator Loop: Decouples rendering and CPU speed from physics"""
        fixed_dt = 1.0 / 100.0  # 10ms steps
        accumulator = 0.0
        last_time = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            frame_time = now - last_time
            last_time = now

            # Avoid the 'Spiral of Death' (cap the max frame time)
            if frame_time > 0.25:
                frame_time = 0.25

            accumulator += frame_time

            # Consume the accumulator in fixed chunks
            while accumulator >= fixed_dt:
                self.step_physics(fixed_dt)
                accumulator -= fixed_dt

            self.alpha = accumulator / fixed_dt

            # Yield to OS - helps No-GIL threads breathe
            time.sleep(0.001)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()


class Renderer:
    """It renders. ZenGL magic."""
    def __init__(self, ctx, platforms):
        self.ctx = ctx
        self.platforms = platforms
        self.image = ctx.image((800, 800))
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4'))

        # Shader now takes a 'camera' uniform to offset the world
        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 450 core
                layout (location = 0) in vec2 in_vert;
                uniform vec4 transform;
                uniform vec2 camera;
                void main() {
                    vec2 pos = (in_vert * transform.zw + transform.xy) - camera;
                    gl_Position = vec4(pos, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 450 core
                layout (location = 0) out vec4 out_color;
                uniform vec3 color;
                void main() { out_color = vec4(color, 1.0); }
            ''',
            framebuffer=[self.image],
            topology='triangle_strip',
            vertex_buffers=[*zengl.bind(self.vbo, '2f', 0)],
            uniforms={
                'transform': [0.0] * 4,
                'color': [1.0] * 3,
                'camera': [0.0, 0.0]
            },
            vertex_count=4,
        )
        self.u_trans = self.pipeline.uniforms['transform']
        self.u_color = self.pipeline.uniforms['color']
        self.u_camera = self.pipeline.uniforms['camera']

        self.stars = [
            (np.random.uniform(-5, 5), np.random.uniform(-0.5, 0.5), 0.01, 0.01)
            for _ in range(50)
        ]

    def draw(self, player_pos):
        self.ctx.new_frame()
        self.image.clear()

        # --- LAYER 1: The Distant Stars (Slow Movement) ---
        self.u_camera[:] = struct.pack('2f', player_pos[0] * 0.2, 0.0)  # 20% speed
        self.u_color[:] = struct.pack('3f', 0.2, 0.2, 0.4)  # Dim Blue

        for sx, sy, sw, sh in self.stars:
            self.u_trans[:] = struct.pack('4f', sx, sy, sw, sh)
            self.pipeline.render()

        # --- LAYER 2: The Game World (Normal Speed) ---
        self.u_camera[:] = struct.pack('2f', player_pos[0], 0.0)  # 100% speed

        # Render Platforms...
        self.u_color[:] = struct.pack('3f', 0.4, 0.4, 0.4)
        for p in self.platforms:
            self.u_trans[:] = struct.pack('4f', p.x, p.y, p.hw, p.hh)
            self.pipeline.render()

        # Render Player...
        self.u_color[:] = struct.pack('3f', 1.0, 0.8, 0.2)
        self.u_trans[:] = struct.pack('4f', player_pos[0], player_pos[1], 0.05, 0.05)
        self.pipeline.render()

        self.image.blit()
        self.ctx.end_frame()


class Game:
    """Finally, the game manager. Glues things together while managing user input."""
    def __init__(self):
        glfw.init()
        self.window = glfw.create_window(800, 800, "ZenGL Scrolling Platformer", None, None)
        glfw.make_context_current(self.window)

        # Add more platforms to test scrolling
        # x, y, half_width, half_height
        platforms = [
            Platform(0.0, -0.8, 3.0, 0.1),
            Platform(0.4, -0.4, 0.2, 0.05),
            Platform(1.2, -0.2, 0.2, 0.05),  # Far right platform
            Platform(2.0, 0.1, 0.2, 0.05),  # Even further
            Platform(3.0, 0.3, 0.2, 0.05),
        ]
        self.physics = PhysicsEngine(platforms)
        self.renderer = Renderer(zengl.context(), platforms)

        glfw.set_key_callback(self.window, self.on_key)
        self.physics.start()

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.physics.keys[key] = True
        elif action == glfw.RELEASE:
            self.physics.keys[key] = False

    def run(self):
        while not glfw.window_should_close(self.window):
            # LERP: (Current * Alpha) + (Previous * (1.0 - Alpha))
            alpha = self.physics.alpha
            current = self.physics.player_data[:2]
            prev = self.physics.prev_player_pos

            visual_pos = current * alpha + prev * (1.0 - alpha)

            self.renderer.draw(visual_pos)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()


if __name__ == "__main__":
    Game().run()