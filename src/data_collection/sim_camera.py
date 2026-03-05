"""
Virtual camera capture module for IsaacLab simulation environments.

Replaces AutoDataCollector's RealSense D435 hardware camera with a simulated
pinhole camera that captures RGB (and optionally depth) images from the
IsaacLab environment. Runs INSIDE the IsaacLab conda subprocess.

Supports two camera types:
  - ``fixed``: World-frame camera defined by position + look-at target.
  - ``body_mounted``: Camera attached to a robot body (e.g., wrist), defined
    by offset position + quaternion orientation in a specified convention.

Two capture strategies per camera:
  1. IsaacLab Camera sensor (preferred) -- attaches to the env step loop.
  2. omni.replicator fallback -- one-off captures when sensor setup fails.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import CameraConfig

logger = logging.getLogger(__name__)


def _look_at_quaternion(
    eye: list[float], target: list[float], up: list[float] | None = None
) -> tuple[float, float, float, float]:
    """Compute a wxyz quaternion that orients a camera at *eye* to look at *target*.

    Uses a right-handed look-at convention where the camera's forward axis is -Z
    (OpenGL / USD convention).

    Returns:
        (w, x, y, z) quaternion.
    """
    if up is None:
        up = [0.0, 0.0, 1.0]

    eye_np = np.array(eye, dtype=np.float64)
    target_np = np.array(target, dtype=np.float64)
    up_np = np.array(up, dtype=np.float64)

    forward = target_np - eye_np
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    forward /= norm

    right = np.cross(forward, up_np)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # forward is parallel to up -- pick an arbitrary perpendicular
        up_np = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_np)
        right_norm = np.linalg.norm(right)
    right /= right_norm

    true_up = np.cross(right, forward)

    # Camera convention: -Z is forward, X is right, Y is up
    # Build rotation matrix (columns = camera axes in world frame)
    rot = np.zeros((3, 3), dtype=np.float64)
    rot[:, 0] = right
    rot[:, 1] = true_up
    rot[:, 2] = -forward  # camera looks along -Z

    # Rotation matrix -> quaternion (Shepperd method)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    # Normalize
    length = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / length, x / length, y / length, z / length)


class SimCamera:
    """Virtual camera that captures RGB (and optionally depth) from an IsaacLab env.

    Supports two initialization modes:

    1. **CameraConfig-based** (new): Pass a ``CameraConfig`` object that defines
       either a fixed or body-mounted camera with explicit orientation.
    2. **Legacy kwargs** (backward compat): Pass ``camera_position`` and
       ``camera_target`` for a world-frame fixed camera.

    Args:
        env: A ``ManagerBasedRLEnv`` instance (already initialized).
        cam_config: CameraConfig instance (preferred).
        resolution: (width, height) in pixels (legacy, overridden by cam_config).
        camera_position: World-frame position [x, y, z] (legacy).
        camera_target: World-frame look-at point [x, y, z] (legacy).
        prim_path: USD prim path for the camera (legacy).
    """

    # Default camera poses (matching robot_profiles)
    DEFAULT_POSITION = [1.5, 1.2, 1.0]
    DEFAULT_TARGET = [0.25, 0.0, 0.3]

    def __init__(
        self,
        env,
        cam_config: CameraConfig | None = None,
        resolution: tuple[int, int] = (640, 480),
        camera_position: list[float] | None = None,
        camera_target: list[float] | None = None,
        prim_path: str = "/World/DataCollectionCamera",
    ) -> None:
        self._env = env
        self._cam_config = cam_config

        # Resolve env_regex_ns for prim paths (e.g. "/World/envs/env_.*")
        self._env_regex_ns = ""
        if hasattr(env, "scene") and hasattr(env.scene, "env_regex_ns"):
            self._env_regex_ns = env.scene.env_regex_ns

        if cam_config is not None:
            self._width, self._height = cam_config.resolution
            self._cam_type = cam_config.cam_type
            self._cam_name = cam_config.name
            if cam_config.cam_type == "fixed":
                self._position = list(cam_config.position) or self.DEFAULT_POSITION.copy()
                self._target = list(cam_config.target) or self.DEFAULT_TARGET.copy()
                self._prim_path = f"{self._env_regex_ns}/{cam_config.name}Camera"
            else:  # body_mounted
                self._position = []  # not used for body_mounted
                self._target = []
                self._prim_path = prim_path  # will be overridden in setup
        else:
            # Legacy mode
            self._width, self._height = resolution
            self._cam_type = "fixed"
            self._cam_name = "front"
            self._position = camera_position or self.DEFAULT_POSITION.copy()
            self._target = camera_target or self.DEFAULT_TARGET.copy()
            self._prim_path = f"{self._env_regex_ns}/DataCollectionCamera"

        self._camera = None  # IsaacLab Camera sensor
        self._use_replicator = False  # fallback flag
        self._rep_annotator = None  # replicator RGB annotator
        self._rep_depth_annotator = None  # replicator depth annotator
        self._rep_render_product = None

        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Try IsaacLab Camera sensor, fall back to omni.replicator."""
        try:
            if self._cam_type == "body_mounted" and self._cam_config is not None:
                self._setup_body_mounted_camera()
            else:
                self._setup_fixed_camera()
            logger.info(
                "SimCamera[%s]: using IsaacLab Camera sensor at %s",
                self._cam_name, self._prim_path,
            )
        except Exception as exc:
            if self._cam_type == "body_mounted":
                # Body-mounted cameras cannot fall back to replicator
                logger.error(
                    "SimCamera[%s]: body_mounted camera setup failed: %s",
                    self._cam_name, exc,
                )
                raise
            logger.warning(
                "SimCamera[%s]: IsaacLab Camera setup failed (%s). "
                "Falling back to omni.replicator.",
                self._cam_name, exc,
            )
            self._use_replicator = True
            self._setup_replicator_camera()
            logger.info("SimCamera[%s]: using omni.replicator fallback", self._cam_name)

    def _setup_fixed_camera(self) -> None:
        """Create a fixed world-frame IsaacLab Camera sensor.

        Uses position + target -> _look_at_quaternion -> convention="world".
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        quat_wxyz = _look_at_quaternion(self._position, self._target)

        camera_cfg = CameraCfg(
            prim_path=self._prim_path,
            update_period=0.0,  # update every sim step
            height=self._height,
            width=self._width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(self._position),
                rot=quat_wxyz,
                convention="world",
            ),
        )

        self._camera = Camera(camera_cfg)
        # NOTE: Don't call reset() here. Camera sensors use timeline-based
        # initialization (PLAY event callback). If we're created after the
        # simulation is already playing, reset() will fail with
        # "_is_initialized = False". The MultiCameraManager handles
        # re-triggering the PLAY event after all cameras are created.

    def _setup_body_mounted_camera(self) -> None:
        """Create a body-mounted IsaacLab Camera sensor.

        Uses offset_pos + offset_rot (wxyz quaternion) + convention from CameraConfig.
        The prim path is set as a child of the parent body so the camera
        moves with the robot link.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        cfg = self._cam_config
        parent_body = cfg.parent_body
        cam_name = cfg.name

        # Prim path: child of the robot body (resolve env_regex_ns)
        self._prim_path = f"{self._env_regex_ns}/Robot/{parent_body}/{cam_name}_cam"

        camera_cfg = CameraCfg(
            prim_path=self._prim_path,
            update_period=0.0,
            height=self._height,
            width=self._width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),  # shorter range for wrist cam
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(cfg.offset_pos),
                rot=tuple(cfg.offset_rot),
                convention=cfg.convention,
            ),
        )

        self._camera = Camera(camera_cfg)
        # NOTE: Don't call reset() here — see comment in _setup_fixed_camera.

    def _setup_replicator_camera(self) -> None:
        """Set up omni.replicator render product as fallback (fixed cameras only)."""
        import omni.replicator.core as rep

        cam = rep.create.camera(
            position=tuple(self._position),
            look_at=tuple(self._target),
        )

        self._rep_render_product = rep.create.render_product(
            cam, (self._width, self._height)
        )

        self._rep_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rep_annotator.attach([self._rep_render_product])

        try:
            self._rep_depth_annotator = rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane"
            )
            self._rep_depth_annotator.attach([self._rep_render_product])
        except Exception:
            self._rep_depth_annotator = None

    # Alias for backward compatibility (old code calls _setup_isaaclab_camera)
    _setup_isaaclab_camera = _setup_fixed_camera

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Capture an RGB image from the camera.

        Should be called after ``env.step()`` so sensor data is fresh.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, in RGB order.
        """
        if self._use_replicator:
            return self._capture_replicator_rgb()
        return self._capture_sensor_rgb()

    def capture_depth(self) -> np.ndarray | None:
        """Capture a depth image (meters) if available.

        Returns:
            np.ndarray of shape (H, W), dtype float32 in meters, or None.
        """
        if self._use_replicator:
            return self._capture_replicator_depth()
        return self._capture_sensor_depth()

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return (self._width, self._height)

    @property
    def name(self) -> str:
        """Camera name (e.g., 'front', 'wrist', 'top')."""
        return self._cam_name

    # ------------------------------------------------------------------
    # IsaacLab Camera sensor capture
    # ------------------------------------------------------------------

    def _capture_sensor_rgb(self) -> np.ndarray:
        """Read RGB from IsaacLab Camera sensor (env index 0)."""
        self._camera.update(dt=0.0)

        # camera.data.output["rgb"] -> (num_envs, H, W, 4) RGBA tensor
        rgba = self._camera.data.output["rgb"]
        img = rgba[0, :, :, :3]  # first env, drop alpha
        return img.cpu().numpy().astype(np.uint8)

    def _capture_sensor_depth(self) -> np.ndarray | None:
        """Read depth from IsaacLab Camera sensor (env index 0)."""
        self._camera.update(dt=0.0)

        depth_data = self._camera.data.output.get("distance_to_image_plane")
        if depth_data is None:
            return None

        depth = depth_data[0, :, :]  # (H, W)
        return depth.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Replicator fallback capture
    # ------------------------------------------------------------------

    def _capture_replicator_rgb(self) -> np.ndarray:
        """One-shot RGB capture via omni.replicator."""
        import omni.replicator.core as rep

        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

        data = self._rep_annotator.get_data()
        if data is None:
            raise RuntimeError("SimCamera: replicator RGB annotator returned None")

        # Replicator returns (H, W, 4) RGBA uint8
        img = np.array(data, dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def _capture_replicator_depth(self) -> np.ndarray | None:
        """One-shot depth capture via omni.replicator."""
        if self._rep_depth_annotator is None:
            return None

        import omni.replicator.core as rep

        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

        data = self._rep_depth_annotator.get_data()
        if data is None:
            return None
        return np.array(data, dtype=np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def update_pose(
        self, position: list[float], target: list[float]
    ) -> None:
        """Reposition the camera at runtime (fixed cameras only).

        Args:
            position: New world-frame position [x, y, z].
            target: New world-frame look-at point [x, y, z].
        """
        if self._cam_type == "body_mounted":
            logger.warning("SimCamera[%s]: cannot update_pose on body_mounted camera", self._cam_name)
            return

        self._position = list(position)
        self._target = list(target)

        if self._use_replicator:
            self._setup_replicator_camera()
        else:
            self._update_sensor_pose()

    def _update_sensor_pose(self) -> None:
        """Update IsaacLab Camera sensor pose via USD."""
        try:
            from pxr import Gf, UsdGeom

            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._prim_path)
            if not prim.IsValid():
                logger.warning("SimCamera: prim %s not found for pose update", self._prim_path)
                return

            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            quat_wxyz = _look_at_quaternion(self._position, self._target)
            pos = Gf.Vec3d(*self._position)
            rot = Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))

            xformable.AddTranslateOp().Set(pos)
            xformable.AddOrientOp().Set(rot)
        except Exception as exc:
            logger.warning("SimCamera: USD pose update failed: %s", exc)


class MultiCameraManager:
    """Manages multiple SimCamera instances from robot profile camera configs.

    Creates one SimCamera per CameraConfig entry. Failed cameras are logged
    but do not block initialization of other cameras.

    Args:
        env: A ``ManagerBasedRLEnv`` instance (already initialized).
        cameras_config: ``{name: CameraConfig}`` dict from robot profile.
    """

    def __init__(self, env, cameras_config: dict[str, CameraConfig]) -> None:
        self.cameras: dict[str, SimCamera] = {}
        has_isaaclab_cameras = False
        for name, cfg in cameras_config.items():
            try:
                self.cameras[name] = SimCamera(env, cam_config=cfg)
                if not self.cameras[name]._use_replicator:
                    has_isaaclab_cameras = True
                logger.info("MultiCameraManager: camera '%s' (%s) initialized", name, cfg.cam_type)
            except Exception as e:
                logger.warning("MultiCameraManager: camera '%s' init failed: %s", name, e)

        # IsaacLab Camera sensors use timeline-based init (PLAY event callback).
        # If we created cameras after the simulation was already playing, their
        # _initialize_callback never fired. Camera reset is deferred to the
        # first capture_all() call, which runs after env.reset() has re-fired
        # the PLAY event. This avoids sim.step(render=True) / sim.render()
        # which can hang indefinitely in headless EGL mode (C++ GIL hold).
        self._env = env
        self._has_isaaclab_cameras = has_isaaclab_cameras
        self._cameras_ready = False
        if not has_isaaclab_cameras:
            self._cameras_ready = True  # no IsaacLab cameras, nothing to defer

    def _deferred_camera_init(self) -> None:
        """Deferred camera initialization — called on first capture_all().

        IsaacLab Camera sensors created after sim.play() miss the PLAY event
        and fail to initialize. Calling sim.stop()/sim.play()/sim.render()
        to re-trigger hangs in headless EGL mode (C++ holds GIL).

        Strategy: try reset(). On failure, switch fixed cameras to
        omni.replicator fallback (which has its own rendering pipeline).
        Body-mounted cameras that fail are disabled.
        """
        to_remove = []
        for name, cam in self.cameras.items():
            if cam._camera is not None and not cam._use_replicator:
                try:
                    cam._camera.reset(env_ids=None)
                    logger.info("MultiCameraManager: camera '%s' reset OK", name)
                except Exception as e:
                    logger.warning(
                        "MultiCameraManager: camera '%s' reset failed: %s", name, e
                    )
                    # Switch fixed cameras to replicator fallback
                    if cam._cam_type == "fixed" and cam._position:
                        try:
                            cam._setup_replicator_camera()
                            cam._use_replicator = True
                            cam._camera = None
                            logger.info(
                                "MultiCameraManager: camera '%s' switched to replicator fallback",
                                name,
                            )
                        except Exception as rep_err:
                            logger.warning(
                                "MultiCameraManager: camera '%s' replicator fallback failed: %s",
                                name, rep_err,
                            )
                            to_remove.append(name)
                    else:
                        logger.warning(
                            "MultiCameraManager: camera '%s' (%s) disabled (no fallback)",
                            name, cam._cam_type,
                        )
                        to_remove.append(name)
        for name in to_remove:
            del self.cameras[name]
        self._cameras_ready = True

    def capture_all(self) -> dict[str, np.ndarray | None]:
        """Capture RGB images from all cameras.

        On first call, performs deferred camera initialization: reset all
        IsaacLab Camera sensors so they pick up data from the render buffer.
        This must be called **after** ``env.reset()`` which fires the PLAY
        event that the Camera sensors listen for.

        Returns:
            ``{camera_name: rgb_array}`` dict. Failed captures are ``None``.
        """
        if not self._cameras_ready:
            self._deferred_camera_init()

        images: dict[str, np.ndarray | None] = {}
        for name, cam in self.cameras.items():
            try:
                images[name] = cam.capture()
            except Exception:
                logger.warning("MultiCameraManager: capture failed for camera '%s'", name)
                images[name] = None
        return images

    @property
    def camera_names(self) -> list[str]:
        """List of successfully initialized camera names."""
        return list(self.cameras.keys())

    def __len__(self) -> int:
        return len(self.cameras)

    def __bool__(self) -> bool:
        return len(self.cameras) > 0


# ==================================================================
# Scene-integrated cameras (proper IsaacLab approach)
# ==================================================================


def inject_cameras_into_scene(env_cfg, robot_cfg, camera_names: list[str] | None = None) -> list[str]:
    """Inject CameraCfg attributes into env_cfg.scene BEFORE env construction.

    This is the proper IsaacLab approach: cameras become part of the
    InteractiveSceneCfg so they receive the PLAY event during sim.play()
    and initialize correctly.  No retrigger / fallback needed.

    Args:
        env_cfg: The ManagerBasedRLEnvCfg instance (with .scene attribute).
        robot_cfg: RobotSimConfig with .cameras dict[str, CameraConfig].
        camera_names: Optional list of camera names to inject (e.g., ["front"]).
            If None, all cameras from robot_cfg are injected.

    Returns:
        List of attribute names added to env_cfg.scene (e.g., ["front_cam", "top_cam"]).
    """
    if not robot_cfg.cameras:
        return []

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    # Use {ENV_REGEX_NS} template — resolved by InteractiveScene.__init__()
    env_ns_template = "{ENV_REGEX_NS}"

    attr_names = []
    cameras_to_inject = robot_cfg.cameras
    if camera_names is not None:
        cameras_to_inject = {k: v for k, v in robot_cfg.cameras.items() if k in camera_names}
    for cam_name, cam_cfg in cameras_to_inject.items():
        attr_name = f"{cam_name}_cam"

        if cam_cfg.cam_type == "fixed":
            quat_wxyz = _look_at_quaternion(cam_cfg.position, cam_cfg.target)
            prim_path = f"{env_ns_template}/{cam_name}Camera"
            camera_cfg = CameraCfg(
                prim_path=prim_path,
                update_period=0.0,
                height=cam_cfg.resolution[1],
                width=cam_cfg.resolution[0],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(cam_cfg.position),
                    rot=quat_wxyz,
                    convention="opengl",  # _look_at_quaternion uses -Z forward (OpenGL)
                ),
            )
        elif cam_cfg.cam_type == "body_mounted":
            prim_path = f"{env_ns_template}/Robot/{cam_cfg.parent_body}/{cam_name}_cam"
            camera_cfg = CameraCfg(
                prim_path=prim_path,
                update_period=0.0,
                height=cam_cfg.resolution[1],
                width=cam_cfg.resolution[0],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 2.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(cam_cfg.offset_pos),
                    rot=tuple(cam_cfg.offset_rot),
                    convention=cam_cfg.convention,
                ),
            )
        else:
            logger.warning("inject_cameras_into_scene: unknown cam_type '%s'", cam_cfg.cam_type)
            continue

        setattr(env_cfg.scene, attr_name, camera_cfg)
        attr_names.append(attr_name)
        logger.info("inject_cameras_into_scene: added '%s' (%s) to scene", attr_name, cam_cfg.cam_type)

    return attr_names


class SceneCameraManager:
    """Camera manager that reads from scene-integrated IsaacLab Camera sensors.

    Unlike MultiCameraManager (which creates cameras after env construction),
    this class wraps cameras that were injected into the scene config BEFORE
    env construction.  They are already initialized and ready to capture.

    Args:
        env: ManagerBasedRLEnv instance (already constructed).
        camera_attr_names: Attribute names on env.scene (from inject_cameras_into_scene).
    """

    def __init__(self, env, camera_attr_names: list[str]) -> None:
        self._env = env
        self._cameras: dict[str, object] = {}
        for attr_name in camera_attr_names:
            try:
                cam_sensor = env.scene[attr_name]
                self._cameras[attr_name] = cam_sensor
                logger.info("SceneCameraManager: '%s' ready", attr_name)
            except Exception as e:
                logger.warning("SceneCameraManager: '%s' not found in scene: %s", attr_name, e)

    def capture_all(self) -> dict[str, np.ndarray | None]:
        """Capture RGB images from all scene cameras.

        Returns:
            ``{camera_attr_name: rgb_array}`` dict. Failed captures are ``None``.
        """
        images: dict[str, np.ndarray | None] = {}
        for name, cam_sensor in self._cameras.items():
            try:
                cam_sensor.update(dt=0.0)
                rgba = cam_sensor.data.output["rgb"]
                img = rgba[0, :, :, :3]  # first env, drop alpha
                images[name] = img.cpu().numpy().astype(np.uint8)
            except Exception as e:
                logger.warning("SceneCameraManager: capture failed for '%s': %s", name, e)
                images[name] = None
        return images

    @property
    def camera_names(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def cameras(self) -> dict[str, object]:
        """Direct access to camera sensors (for _capture_front_image helper)."""
        return self._cameras

    def __len__(self) -> int:
        return len(self._cameras)

    def __bool__(self) -> bool:
        return len(self._cameras) > 0
