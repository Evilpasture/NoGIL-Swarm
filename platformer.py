import glfw
import zengl
import numpy as np
import threading
import time
import struct

from dataclasses import dataclass

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
        self.platforms = platforms
        self.pw, self.ph = 0.05, 0.05
        self.keys = {}
        self.running = True

    def step_physics(self, dt):
        """Helper to contain the actual math, making it independent of loop speed"""
        px, py, vx, vy = self.player_data

        # 1. Scale constants for 'Per-Second' logic
        # Gravity is now roughly -2.2 units per second squared
        gravity = -0.22 * dt
        friction = 0.85 ** (dt * 100)

        vy += gravity
        vx *= friction

        if self.keys.get(glfw.KEY_LEFT): vx -= 0.0025
        if self.keys.get(glfw.KEY_RIGHT): vx += 0.0025

        new_x, new_y = px + vx, py + vy
        on_ground = False

        for plat in self.platforms:
            rx, ry, rw, rh = plat.x, plat.y, plat.hw, plat.hh

            if (abs(new_x - rx) < (rw + self.pw)) and (abs(new_y - ry) < (rh + self.ph)):
                # Landing/Ceiling
                if abs(px - rx) < (rw + self.pw - 0.02):
                    if vy < 0 and py >= (ry + rh - 0.02):
                        new_y, vy, on_ground = ry + rh + self.ph, 0, True
                    elif vy > 0 and py <= (ry - rh + 0.02):
                        new_y, vy = ry - rh - self.ph, 0
                # Walls
                elif abs(py - ry) < (rh + self.ph - 0.02):
                    new_x, vx = (rx - rw - self.pw if vx > 0 else rx + rw + self.pw), 0

        if self.keys.get(glfw.KEY_UP) and on_ground:
            vy = 0.045  # Back to your original perfect jump value

        self.player_data[:] = [new_x, new_y, vx, vy]

    def update(self):
        """The Loop: Manages time and calls step_physics"""
        target_dt = 1.0 / 100.0  # 100Hz Physics
        last_time = time.perf_counter()
        while self.running:
            while time.perf_counter() - last_time < target_dt:
                pass

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # This ensures that even if the thread sleeps longer,
            # the player doesn't move 50x faster than intended.
            self.step_physics(dt)


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

    def draw(self, player_pos):
        self.ctx.new_frame()
        self.image.clear()

        # Update camera to follow player (Lerp for smoothness if desired)
        # We only follow X here, but you can follow Y too.
        self.u_camera[:] = struct.pack('2f', player_pos[0], 0.0)

        # Platforms (Rendered relative to camera)
        self.u_color[:] = struct.pack('3f', 0.4, 0.4, 0.4)
        for p in self.platforms:
            self.u_trans[:] = struct.pack(
                '4f',
                p.x, p.y, p.hw, p.hh
            )
            self.pipeline.render()

        # Player (Will appear centered because camera = player_pos)
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
            Platform(0.0, -0.8, 1.0, 0.1),
            Platform(0.4, -0.4, 0.2, 0.05),
            Platform(1.2, -0.2, 0.2, 0.05),  # Far right platform
            Platform(2.0, 0.1, 0.2, 0.05),  # Even further
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
            self.renderer.draw(self.physics.player_data[:2])
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()


if __name__ == "__main__":
    Game().run()