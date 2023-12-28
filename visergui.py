from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from viser_utils import qvec2rotmat
import cv2
from viser_utils import Timer
from collections import deque
from gs_renderer_4d import Renderer, MiniCam
from PIL import Image



def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port
        self.server = viser.ViserServer(port=self.port)

        self.need_update = False


        self.timestep_slider = self.server.add_gui_slider(
            "Timestep", min=0, max=13, step=0.1, initial_value=0
        )

        @self.timestep_slider.on_update
        def _(_):
            self.need_update = True
                
        self.c2ws = []
        self.camera_infos = []


        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def set_renderer(self, renderer, fixed_cam):
        self.renderer = renderer
        self.fixed_cam = fixed_cam

    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            for client in self.server.get_clients().values():
                camera = client.camera
                c2w = get_c2w(camera)
                try:
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    cur_cam = MiniCam(
                        c2w,
                        self.fixed_cam.image_width,
                        self.fixed_cam.image_height,
                        self.fixed_cam.FoVy,
                        self.fixed_cam.FoVx,
                        self.fixed_cam.znear,
                        self.fixed_cam.zfar,
                        gs_convention=False,
                        time=self.timestep_slider.value
                    )

                    outputs = self.renderer.render(cur_cam)
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.

                    # out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                    out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                    out = np.transpose(out, (1, 2, 0))

                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue
                client.set_background_image(out, format="jpeg")
                self.debug_idx += 1