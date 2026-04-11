"""Smoke tests for the UroSim gRPC client scaffold.

These tests only verify imports, method presence, and protobuf message
instantiation. Actual RPC round-trips require a live Unity simulator and
live behind ``@pytest.mark.unity``.
"""

from __future__ import annotations

import numpy as np
import pytest

from urosim.client import UroSimClient
from urosim.client.generated import urosim_pb2


def test_client_importable_and_has_methods() -> None:
    """UroSimClient instantiates without RPC and exposes every documented method."""
    # grpc.insecure_channel is lazy — instantiation must not raise even if
    # no server is listening on localhost:50051.
    client = UroSimClient("localhost:50051")
    try:
        for name in (
            "reset",
            "step",
            "load_kidney",
            "get_topology_graph",
            "set_config",
            "set_visual_params",
            "start_recording",
            "stop_recording",
            "close",
        ):
            assert callable(getattr(client, name)), f"missing method: {name}"
    finally:
        client.close()


def test_client_context_manager() -> None:
    """UroSimClient works as a context manager and closes the channel on exit."""
    with UroSimClient("localhost:50051") as client:
        assert isinstance(client, UroSimClient)


def test_proto_messages_instantiable() -> None:
    """Every protobuf message the client constructs must round-trip its fields."""
    cmd = urosim_pb2.ScopeCommand(
        advance_mm=1.0, roll_deg=2.0, kappa=0.1, phi=0.5, laser_active=False
    )
    assert cmd.advance_mm == pytest.approx(1.0)
    assert cmd.phi == pytest.approx(0.5)
    assert cmd.laser_active is False

    pose = urosim_pb2.Pose(x=1.0, y=2.0, z=3.0, qw=1.0)
    assert pose.z == pytest.approx(3.0)
    assert pose.qw == pytest.approx(1.0)

    tel = urosim_pb2.Telemetry(
        advance_position_mm=10.0,
        roll_position_deg=0.0,
        kappa_actual_deg=5.0,
        phi_actual_deg=90.0,
        tip_contact_force_N=0.25,
        laser_active=True,
        perforation=False,
    )
    assert tel.tip_contact_force_N == pytest.approx(0.25)
    assert tel.laser_active is True

    cov = urosim_pb2.CoverageState(
        visited_calyx_ids=["mc_0", "mc_1"],
        total_calyces=8,
        coverage_fraction=0.25,
    )
    cov.calyx_observation_pct["mc_0"] = 0.5
    assert cov.total_calyces == 8
    assert list(cov.visited_calyx_ids) == ["mc_0", "mc_1"]
    assert cov.calyx_observation_pct["mc_0"] == pytest.approx(0.5)

    stone_state = urosim_pb2.StoneState(
        index=0,
        pose=urosim_pb2.Pose(qw=1.0),
        hp_fraction=0.8,
        composition="CaOx_mono",
        fragmented=False,
        fragment_count=0,
    )
    assert stone_state.composition == "CaOx_mono"
    assert stone_state.hp_fraction == pytest.approx(0.8)

    step = urosim_pb2.StepResult(
        tip_pose_gt=urosim_pb2.Pose(qw=1.0),
        tip_pose_reported=urosim_pb2.Pose(qw=1.0),
        telemetry=tel,
        coverage=cov,
        stones=[stone_state],
        timestamp=12.5,
        frame_number=42,
        rgb_jpeg=b"",
    )
    assert step.frame_number == 42
    assert len(step.stones) == 1

    obs = urosim_pb2.Observation(
        tip_pose_gt=urosim_pb2.Pose(qw=1.0),
        tip_pose_reported=urosim_pb2.Pose(qw=1.0),
        telemetry=tel,
        coverage=cov,
        timestamp=0.0,
        frame_number=0,
    )
    assert obs.telemetry.laser_active is True


def test_kidney_data_message_instantiable() -> None:
    """KidneyData accepts the exact payload types UroSimClient produces."""
    stone = urosim_pb2.StoneSpec(
        position=urosim_pb2.Pose(x=0.0, y=0.0, z=0.0, qw=1.0),
        radius_mm=2.5,
        composition="CaOx_mono",
        hp=100.0,
    )
    assert stone.composition == "CaOx_mono"
    assert stone.radius_mm == pytest.approx(2.5)

    kidney = urosim_pb2.KidneyData(
        mesh_obj=b"o cube\nv 0 0 0\n",
        normals_bin=np.zeros((4, 3), dtype=np.float32).tobytes(),
        graph_json=b"{}",
        stones=[stone],
        ureter_entry=urosim_pb2.Pose(x=1.0, y=2.0, z=3.0, qw=1.0),
        texture_coords_bin=np.zeros((4, 3), dtype=np.float32).tobytes(),
        coverage_points_json=b"{}",
    )
    assert len(kidney.stones) == 1
    assert kidney.ureter_entry.x == pytest.approx(1.0)
    # 4 vertices * 3 coords * 4 bytes = 48 bytes per blob.
    assert len(kidney.normals_bin) == 48
    assert len(kidney.texture_coords_bin) == 48


def test_config_visual_and_recording_messages() -> None:
    cfg = urosim_pb2.SimConfig(dt=0.01, dynamics_enabled=True)
    assert cfg.dt == pytest.approx(0.01)
    assert cfg.dynamics_enabled is True

    vis = urosim_pb2.VisualParams(
        tissue_hue_offset=0.1,
        tissue_saturation_scale=1.2,
        light_intensity_scale=0.9,
        light_temperature_K=5500.0,
        turbidity=0.3,
        vein_density_scale=1.0,
    )
    assert vis.light_temperature_K == pytest.approx(5500.0)

    rec_cfg = urosim_pb2.RecordingConfig(
        output_dir="/tmp/out",
        save_rgb=True,
        save_depth=False,
        save_flow=False,
        save_instance_seg=True,
        image_width=1280,
        image_height=720,
        pose_format="quat",
    )
    assert rec_cfg.image_width == 1280
    assert rec_cfg.pose_format == "quat"

    rec_res = urosim_pb2.RecordingResult(
        output_dir="/tmp/out",
        frames_recorded=120,
        duration_seconds=4.0,
    )
    assert rec_res.frames_recorded == 120


def test_topology_graph_message() -> None:
    calyx = urosim_pb2.CalyxInfo(
        id="mc_0",
        center=urosim_pb2.Pose(x=1.0, y=2.0, z=3.0, qw=1.0),
        radius_mm=3.5,
        type="minor_calyx",
    )
    topo = urosim_pb2.TopologyGraph(
        graph_json=b'{"nodes":[],"edges":[]}',
        calyces=[calyx],
    )
    assert len(topo.calyces) == 1
    assert topo.calyces[0].type == "minor_calyx"


@pytest.mark.unity
def test_reset_live() -> None:
    """Round-trip ``reset`` against a live Unity simulator on localhost:50051."""
    with UroSimClient() as client:
        obs = client.reset(seed=42)
    assert "tip_pose_gt" in obs
    assert obs["tip_pose_gt"].shape == (7,)
