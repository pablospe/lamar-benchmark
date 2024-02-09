import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from scantools import (
    logger,
    run_meshing,
    run_navvis_to_capture,
    run_rendering,
    to_meshlab_visualization,
)
from scantools.capture import Capture
from scantools.run_qrcode_detection import run_qrcode_detection_session
from scantools.scanners.navvis.camera_tiles import TileFormat

TILE_CHOICES = sorted([attr.name.split("_")[1] for attr in TileFormat])

description = """
ToDo...
"""


def run(
    input_path: Path,
    output_path: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
    tiles_format: Optional[str] = "3x3",
    meshing_method: str = "advancing_front",
    qrcode_detection: bool = True,
    use_simplified_mesh: bool = False,
    visualization: bool = True,
    **kargs,
):
    capture_path = output_path or Path.cwd()

    if capture_path.exists():
        capture = Capture.load(capture_path)
    else:
        capture_path.mkdir(exist_ok=True, parents=True)
        capture = Capture(sessions={}, path=capture_path)

    mesh_id = "mesh"
    if use_simplified_mesh:
        mesh_id += "_simplified"

    # If `sessions` is not provided, run for all sessions in the `input_path`.
    if sessions is None:
        sessions = [p.name for p in input_path.iterdir() if p.is_dir()]

    for session in sessions:
        navvis_path = input_path / session
        if session not in capture.sessions:
            logger.info("Exporting NavVis session %s.", session)
            run_navvis_to_capture.run(
                navvis_path,
                capture,
                tiles_format,
                session,
                export_as_rig=True,
                copy_pointcloud=True,
            )

        if (
            not capture.sessions[session].proc
            or mesh_id not in capture.sessions[session].proc.meshes
        ):
            logger.info("Meshing session %s.", session)
            run_meshing.run(
                capture,
                session,
                "point_cloud_final",
                mesh_id,
                method=meshing_method,
            )

        if not capture.sessions[session].depths:
            logger.info("Rendering session %s.", session)
            run_rendering.run(capture, session, mesh_id=mesh_id)

        if qrcode_detection:
            capture_qrcode_path = Path(capture_path.as_posix() + "-qrcode")
            capture_qrcode_path.mkdir(exist_ok=True, parents=True)
            capture_qrcode = Capture(sessions={}, path=capture_qrcode_path)

            # Copy mesh from capture_path to capture_qrcode_path.
            shutil.copytree(
                str(capture.proc_path(session) / "meshes"),
                str(capture_qrcode.proc_path(session) / "meshes"),
                dirs_exist_ok=True,
            )

            # QR codes are captured intentionally by approaching them closely.
            # As a result, we don't use tiling, which could potentially split
            # the QR code across multiple tiles.
            tiles_format_qrcode = "none"
            run_navvis_to_capture.run(
                navvis_path, capture_qrcode, tiles_format_qrcode
            )

            # Reload capture_qrcode, so we have access to the copied meshes.
            capture_qrcode = Capture.load(capture_qrcode_path)

            # Detect QR codes in the session.
            run_qrcode_detection_session(
                capture_qrcode, session, mesh_id, **kargs
            )

            # Copy QR code output from capture_qrcode to capture.
            shutil.copytree(
                str(capture_qrcode.proc_path(session) / "qrcodes"),
                str(capture.proc_path(session) / "qrcodes"),
                dirs_exist_ok=True,
            )

        if visualization:
            to_meshlab_visualization.run(
                capture,
                session,
                f"trajectory_{session}",
                export_mesh=True,
                export_poses=True,
                mesh_id=mesh_id,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Specifies NavVis data path. Inside this path, there should be "
        "a folder for each session. Each session folder should be the output "
        "of the NavVis data conversion tool (proc/ folders).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=False,
        default=None,
        help="Output path. Default: current working directory. If the path "
        "does not exist, it will be created.",
    )
    parser.add_argument(
        "--tiles_format",
        type=str,
        required=False,
        default="3x3",
        choices=TILE_CHOICES,
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        type=str,
        required=False,
        default=None,
        help="[Optional] List of sessions to process. Useful when "
        "processing only specific sessions.",
    )
    parser.add_argument(
        "--use_simplified_mesh",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Use simplified mesh. Default: False. Pass --use_simplified_mesh "
        "to set to True. This is useful for large scenes.",
    )
    parser.add_argument(
        "--meshing_method",
        type=str,
        required=False,
        default="advancing_front",
        choices=["advancing_front", "poisson"],
        help="Meshing method. Default: advancing_front.",
    )
    parser.add_argument(
        "--qrcode_detection",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
        help="Run QR code detection. Default: True. ",
    )
    parser.add_argument(
        "--visualization",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
        help="Write out MeshLab visualization. Default: True. "
        "Pass --no-visualization to set to False.",
    )
    parser.add_argument(
        "--txt_format",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
        help="Write out QR maps in txt format. Default: True.",
    )
    parser.add_argument(
        "--json_format",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Write out QR maps in json format. Default: False.",
    )
    args = parser.parse_args().__dict__

    run(**args)