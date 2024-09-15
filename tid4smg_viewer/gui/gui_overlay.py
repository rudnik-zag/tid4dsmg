import torch
import imgui

from imgui.integrations.pyglet import PygletRenderer

import OpenGL.GL as gl


class GUIOverlay:
    def __init__(self, window, components, logic):
        imgui.create_context()
        self.impl = PygletRenderer(window)
        self.imgui_io = imgui.get_io()

        imgui.new_frame()
        imgui.end_frame()

        self.window = window
        self.make_initial_layout = True

        self.components = {c: True for c in components}
        self.draw_bounding_volume = False
        # self.components["gaussians"] = True

        # self.background_color = (0.0, 177.0 / 255.0, 64 / 255.0)
        self.background_color = (0.0, 0.0, 0.0)
        self.background_changed = False
        self.FOV = 30
        self.focal = 3000.0
        self.fov_changed = False

        self.mesh_opacity = 0.0
        self.point_opacity = 0.0

        self.logic = logic
        self.controls = torch.zeros(self.logic.gui_control_count, device=self.logic.device)
        self.control_names = self.logic.gui_control_names
        self.control_names = ["_".join(name.split("_")[1:]) for name in self.control_names]
        self.control_filter = ""
        self.active_control_slides = [True] * len(self.control_names)

        self.gui_min = self.logic.gui_min.cpu().numpy()
        self.gui_max = self.logic.gui_max.cpu().numpy()

        self.play = False
        self.recording = False
        self.frame_idx = 0

        self.control_changed = False

    def render_settings(self):
        imgui.set_next_window_size(190, 400)
        imgui.set_next_window_position(0.5, 0.5)
        imgui.set_next_window_bg_alpha(1.0)

        imgui.begin("Settings")
        imgui.begin_child("Components", width=-5, height=200, border=True)
        for c in self.components:
            changed, self.components[c] = imgui.checkbox(c, self.components[c])
            if changed:
                self.window.canvas.needs_update = True

        changed, self.draw_bounding_volume = imgui.checkbox("AABB", self.draw_bounding_volume)

        imgui.push_item_width(-100)
        _, self.mesh_opacity = imgui.slider_float(
            "Mesh opacity", self.mesh_opacity, min_value=0.0, max_value=1.0
        )

        _, self.point_opacity = imgui.slider_float(
            "Point opacity", self.point_opacity, min_value=0.0, max_value=1.0
        )

        imgui.end_child()

        imgui.begin_child("View", width=-5, border=True)
        self.background_changed, self.background_color = imgui.color_edit3(
            "", *self.background_color
        )

        if self.background_changed:
            self.window.canvas.needs_update = True

        self.fov_changed, self.focal = imgui.slider_int(
            "focal", self.focal, min_value=50, max_value=5000
        )

        record_clicked = imgui.button("Stop recording" if self.recording else "Start recording")
        if record_clicked:
            self.recording = not self.recording

        imgui.end_child()

        imgui.end()

    def render_playback(self):
        imgui.set_next_window_size(self.window.width, 35)
        imgui.set_next_window_position(0, self.window.height - 35)
        imgui.set_next_window_bg_alpha(1.0)

        imgui.begin("Playback", flags=imgui.WINDOW_NO_TITLE_BAR)

        play_clicked = imgui.button("||" if self.play else ">")
        if play_clicked:
            self.play = not self.play

        imgui.same_line()
        changed, self.frame_idx = imgui.slider_int(
            "Frame", self.frame_idx, min_value=0, max_value=len(self.window.animation) - 1
        )

        if changed or self.play:
            self.controls = self.window.animation[self.frame_idx]
            self.control_changed = True

        # imgui.same_line()

        imgui.end()

        if self.play:
            self.frame_idx += 1
            self.frame_idx = max(0, min(self.frame_idx, len(self.window.animation) - 1))

    def render(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        imgui.new_frame()

        self.render_settings()
        self.render_controls()

        if self.window.qsa_path is not None:
            self.render_playback()

        imgui.end_frame()
