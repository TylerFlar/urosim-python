"""Thin gRPC client wrapper around the generated UroSim stubs.

This module is the **only** place in the repo allowed to import from
:mod:`urosim.client.generated`. Higher-level modules (``env/``, ``agents/``,
``recording/``) should depend on :class:`UroSimClient` rather than touching
protobuf directly.

The client converts between Python-friendly types (``numpy.ndarray``,
``dict``, ``networkx.DiGraph``) and the protobuf messages defined in
``protos/urosim.proto``. It does **not** depend on the anatomy generation
pipeline (``urosim.anatomy.generator`` / ``urosim.anatomy.topology``); it
only knows how to serialize a :class:`~urosim.anatomy.kidney_model.KidneyModel`
that has already been built elsewhere.

Note: constructing a :class:`UroSimClient` does not attempt any RPC — the
underlying gRPC channel is lazy. Every method will raise if no Unity server
is listening at the configured address.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import TYPE_CHECKING, Any

import grpc
import networkx as nx
import numpy as np

from urosim.anatomy.kidney_model import _graph_to_jsonable
from urosim.client.generated import urosim_pb2, urosim_pb2_grpc

if TYPE_CHECKING:
    from urosim.anatomy.kidney_model import KidneyModel


class UroSimClient:
    """gRPC client for the Unity-side UroSim service.

    Args:
        address: ``host:port`` string for the Unity gRPC server. Defaults
            to ``"localhost:50051"``.
    """

    def __init__(self, address: str = "localhost:50051") -> None:
        self._address = address
        self._channel = grpc.insecure_channel(address)
        self._stub = urosim_pb2_grpc.UroSimStub(self._channel)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> UroSimClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # RPCs
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int = 0,
        preset: str = "",
        fidelity: str = "full",
        camera_profile: str = "",
    ) -> dict[str, Any]:
        """Reset the simulator and return an initial observation.

        Args:
            seed: Integer seed for any RNG-driven reset behavior.
            preset: Optional named config preset on the Unity side.
            fidelity: Fidelity tier, e.g. ``"full"`` or ``"lite"``.
            camera_profile: Optional named camera profile.

        Returns:
            Observation dict with the same layout as :meth:`step`.
        """
        req = urosim_pb2.ResetRequest(
            seed=seed,
            config_preset=preset,
            fidelity=fidelity,
            camera_profile=camera_profile,
        )
        return _step_like_to_dict(self._stub.Reset(req))

    def step(
        self, cmd: dict[str, Any] | urosim_pb2.ScopeCommand
    ) -> dict[str, Any]:
        """Advance the simulator one step with the given scope command.

        Args:
            cmd: Either a ``ScopeCommand`` protobuf message, or a dict of
                keyword arguments suitable for constructing one
                (``advance_mm``, ``roll_deg``, ``kappa``, ``phi``,
                ``laser_active``).

        Returns:
            Dict with keys ``tip_pose_gt``, ``tip_pose_reported``
            (both ``(7,)`` float64 arrays), ``telemetry`` (dict),
            ``coverage`` (dict), ``stones`` (list of dicts),
            ``timestamp`` (float), ``frame_number`` (int), ``rgb_jpeg``
            (bytes).
        """
        if isinstance(cmd, dict):
            cmd = urosim_pb2.ScopeCommand(**cmd)
        return _step_like_to_dict(self._stub.Step(cmd))

    def load_kidney(self, kidney: KidneyModel) -> None:
        """Upload a :class:`KidneyModel` to the Unity simulator.

        Raises:
            RuntimeError: If the server returns a non-success status.
        """
        req = _kidney_model_to_proto(kidney)
        status = self._stub.LoadKidney(req)
        if not status.success:
            raise RuntimeError(f"LoadKidney failed: {status.message}")

    def get_topology_graph(self) -> nx.DiGraph:
        """Fetch the loaded kidney's topology graph as a ``networkx.DiGraph``."""
        resp = self._stub.GetTopologyGraph(urosim_pb2.Empty())
        data = json.loads(resp.graph_json.decode("utf-8"))
        return nx.node_link_graph(data, directed=True, edges="edges")

    def set_config(self, dt: float, dynamics_enabled: bool = True) -> None:
        """Configure sim timestep and dynamics toggles.

        Raises:
            RuntimeError: If the server returns a non-success status.
        """
        status = self._stub.SetConfig(
            urosim_pb2.SimConfig(dt=dt, dynamics_enabled=dynamics_enabled)
        )
        if not status.success:
            raise RuntimeError(f"SetConfig failed: {status.message}")

    def set_visual_params(self, **kwargs: float) -> None:
        """Adjust tissue / lighting / turbidity / vein visual parameters.

        Keyword arguments map directly onto ``VisualParams`` fields:
        ``tissue_hue_offset``, ``tissue_saturation_scale``,
        ``light_intensity_scale``, ``light_temperature_K``, ``turbidity``,
        ``vein_density_scale``.

        Raises:
            RuntimeError: If the server returns a non-success status.
        """
        status = self._stub.SetVisualParams(urosim_pb2.VisualParams(**kwargs))
        if not status.success:
            raise RuntimeError(f"SetVisualParams failed: {status.message}")

    def start_recording(self, output_dir: str, **kwargs: Any) -> None:
        """Start recording session to ``output_dir``.

        Keyword arguments map onto ``RecordingConfig`` fields (``save_rgb``,
        ``save_depth``, ``save_flow``, ``save_instance_seg``, ``image_width``,
        ``image_height``, ``pose_format``).

        Raises:
            RuntimeError: If the server returns a non-success status.
        """
        cfg = urosim_pb2.RecordingConfig(output_dir=output_dir, **kwargs)
        status = self._stub.StartRecording(cfg)
        if not status.success:
            raise RuntimeError(f"StartRecording failed: {status.message}")

    def stop_recording(self) -> dict[str, Any]:
        """Stop the active recording session and return its summary."""
        resp = self._stub.StopRecording(urosim_pb2.Empty())
        return {
            "output_dir": resp.output_dir,
            "frames_recorded": int(resp.frames_recorded),
            "duration_seconds": float(resp.duration_seconds),
        }


# ----------------------------------------------------------------------
# Conversion helpers (module-private)
# ----------------------------------------------------------------------


def _pose_to_array(pose: urosim_pb2.Pose) -> np.ndarray:
    """Convert a ``Pose`` message to a ``(7,)`` float64 array."""
    return np.array(
        [pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz, pose.qw],
        dtype=np.float64,
    )


def _xyz_to_pose(arr: np.ndarray) -> urosim_pb2.Pose:
    """Build a position-only ``Pose`` from an ``(x, y, z)`` array.

    Sets ``qw = 1.0`` so the quaternion is the identity rotation.
    """
    xyz = np.asarray(arr, dtype=np.float64).reshape(-1)
    if xyz.size < 3:
        raise ValueError(f"expected xyz of length >=3, got {xyz.size}")
    return urosim_pb2.Pose(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]), qw=1.0)


def _telemetry_to_dict(tel: urosim_pb2.Telemetry) -> dict[str, Any]:
    return {
        "advance_position_mm": float(tel.advance_position_mm),
        "roll_position_deg": float(tel.roll_position_deg),
        "kappa_actual_deg": float(tel.kappa_actual_deg),
        "phi_actual_deg": float(tel.phi_actual_deg),
        "tip_contact_force_N": float(tel.tip_contact_force_N),
        "laser_active": bool(tel.laser_active),
        "perforation": bool(tel.perforation),
    }


def _coverage_to_dict(cov: urosim_pb2.CoverageState) -> dict[str, Any]:
    return {
        "visited_calyx_ids": list(cov.visited_calyx_ids),
        "total_calyces": int(cov.total_calyces),
        "coverage_fraction": float(cov.coverage_fraction),
        "calyx_observation_pct": {k: float(v) for k, v in cov.calyx_observation_pct.items()},
    }


def _stone_state_to_dict(stone: urosim_pb2.StoneState) -> dict[str, Any]:
    return {
        "index": int(stone.index),
        "pose": _pose_to_array(stone.pose),
        "hp_fraction": float(stone.hp_fraction),
        "composition": str(stone.composition),
        "fragmented": bool(stone.fragmented),
        "fragment_count": int(stone.fragment_count),
    }


def _step_like_to_dict(
    msg: urosim_pb2.StepResult | urosim_pb2.Observation,
) -> dict[str, Any]:
    """Decode a ``StepResult`` or ``Observation`` into Python-friendly types.

    ``StepResult`` and ``Observation`` share an identical field layout in
    the proto schema, so a single decoder handles both.
    """
    return {
        "tip_pose_gt": _pose_to_array(msg.tip_pose_gt),
        "tip_pose_reported": _pose_to_array(msg.tip_pose_reported),
        "telemetry": _telemetry_to_dict(msg.telemetry),
        "coverage": _coverage_to_dict(msg.coverage),
        "stones": [_stone_state_to_dict(s) for s in msg.stones],
        "timestamp": float(msg.timestamp),
        "frame_number": int(msg.frame_number),
        "rgb_jpeg": bytes(msg.rgb_jpeg),
    }


def _mesh_to_obj_bytes(mesh: Any) -> bytes:
    """Export a ``trimesh.Trimesh`` to Wavefront OBJ bytes.

    ``trimesh.Trimesh.export(file_type='obj')`` returns ``str``; older /
    some backends may return ``bytes``. Handle both.
    """
    exported = mesh.export(file_type="obj")
    if isinstance(exported, bytes):
        return exported
    return exported.encode("utf-8")


def _stone_spec_to_proto(stone: Any) -> urosim_pb2.StoneSpec:
    return urosim_pb2.StoneSpec(
        position=_xyz_to_pose(stone.position),
        radius_mm=float(stone.radius_mm),
        composition=str(stone.composition),
        hp=float(stone.hp),
    )


def _kidney_model_to_proto(kidney: KidneyModel) -> urosim_pb2.KidneyData:
    """Serialize a :class:`KidneyModel` to a ``KidneyData`` protobuf.

    Binary arrays (``normals``, ``texture_coords``) are packed as
    C-contiguous float32 blobs — halved from the native float64 layout
    to keep the gRPC payload small. The Unity side reinterprets them as
    ``(V, 3)`` where ``V`` matches the vertex count of ``mesh_obj``.
    """
    normals = np.ascontiguousarray(kidney.normals, dtype=np.float32)
    texcoords = np.ascontiguousarray(kidney.texture_coords, dtype=np.float32)

    coverage_doc = {
        k: np.asarray(v, dtype=np.float64).tolist()
        for k, v in kidney.calyx_coverage_points.items()
    }

    return urosim_pb2.KidneyData(
        mesh_obj=_mesh_to_obj_bytes(kidney.mesh),
        normals_bin=normals.tobytes(),
        graph_json=json.dumps(_graph_to_jsonable(kidney.graph)).encode("utf-8"),
        stones=[_stone_spec_to_proto(s) for s in kidney.stones],
        ureter_entry=_xyz_to_pose(kidney.ureter_entry),
        texture_coords_bin=texcoords.tobytes(),
        coverage_points_json=json.dumps(coverage_doc).encode("utf-8"),
    )
