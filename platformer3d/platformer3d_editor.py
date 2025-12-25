import glfw
import numpy as np
import json
from ..platformer3d import Platform, ray_aabb_intersect

class Editor:
    def __init__(self, platforms):
        self.platforms = platforms
        self.selected_index = -1
        self.keys = {}
        # Editor "Free Cam" state
        self.cam_x, self.cam_y, self.cam_z = 0.0, 5.0, 10.0
        self.cam_yaw, self.cam_pitch = -1.57, -0.5

    def update(self, dt):
        speed = 10.0 * dt
        if self.keys.get(glfw.KEY_LEFT_SHIFT): speed = 30.0 * dt
        cx, cz = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        rx, rz = np.cos(self.cam_yaw + np.pi / 2), np.sin(self.cam_yaw + np.pi / 2)

        if self.keys.get(glfw.KEY_W): self.cam_x += cx * speed; self.cam_z += cz * speed
        if self.keys.get(glfw.KEY_S): self.cam_x -= cx * speed; self.cam_z -= cz * speed
        if self.keys.get(glfw.KEY_A): self.cam_x -= rx * speed; self.cam_z -= rz * speed
        if self.keys.get(glfw.KEY_D): self.cam_x += rx * speed; self.cam_z += rz * speed
        if self.keys.get(glfw.KEY_E): self.cam_y += speed
        if self.keys.get(glfw.KEY_Q): self.cam_y -= speed

    def select_object(self):
        # Raycast from camera
        dx = np.cos(self.cam_yaw) * np.cos(self.cam_pitch)
        dy = np.sin(self.cam_pitch)
        dz = np.sin(self.cam_yaw) * np.cos(self.cam_pitch)
        origin = np.array([self.cam_x, self.cam_y, self.cam_z])
        direction = np.array([dx, dy, dz])

        closest, idx = 999.0, -1
        for i, p in enumerate(self.platforms):
            b_min = np.array([p.x - p.hw, p.y - p.hh, p.z - p.hd])
            b_max = np.array([p.x + p.hw, p.y + p.hh, p.z + p.hd])
            dist = ray_aabb_intersect(origin, direction, b_min, b_max)
            if dist != float('inf') and dist < closest:
                closest, idx = dist, i
        self.selected_index = idx

    def handle_input(self, key, action, mods):
        if action == glfw.PRESS:
            # Add Platform
            if key == glfw.KEY_N:
                dist = 5.0
                nx = self.cam_x + np.cos(self.cam_yaw) * dist
                ny = self.cam_y + np.sin(self.cam_pitch) * dist
                nz = self.cam_z + np.sin(self.cam_yaw) * dist
                nx, ny, nz = round(nx * 2) / 2, round(ny * 2) / 2, round(nz * 2) / 2
                self.platforms.append(Platform(nx, ny, nz, 1.0, 0.2, 1.0))
                self.selected_index = len(self.platforms) - 1

            # Delete Platform
            if key == glfw.KEY_DELETE and self.selected_index != -1:
                self.platforms.pop(self.selected_index)
                self.selected_index = -1

            # Save
            if key == glfw.KEY_F5:
                with open('level.json', 'w') as f: json.dump([p.to_dict() for p in self.platforms], f)
                print("Saved level.json")

        # Modification Logic (Move/Resize)
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