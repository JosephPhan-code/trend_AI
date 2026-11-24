"""
Flask Web Server for Testing YOLO26s-P2 INT8 Drone Detection & Tracking on Videos.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import time
from pathlib import Path

os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from inference_engine import (
    JobManager,
    detect_best_video_codec,
    get_available_models,
    get_available_trackers,
    get_hardware_info,
    load_quantization_report,
)

BASE_WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_WEB_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

manager = JobManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def api_info():
    fourcc, ext, mime, playable = detect_best_video_codec()
    return jsonify({
        "models": get_available_models(),
        "trackers": get_available_trackers(),
        "quantization_report": load_quantization_report(),
        "hardware": get_hardware_info(),
        "codec": {
            "fourcc": fourcc,
            "ext": ext,
            "mime": mime,
            "browser_playable": playable,
        },
        "jobs": manager.list_jobs(limit=25),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "Thiếu trường 'file' khi upload video."}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Chưa chọn file video."}), 400

    orig_name = Path(f.filename).name
    safe_name = secure_filename(orig_name) or ("upload" + Path(orig_name).suffix.lower())
    try:
        job = manager.create_uploaded_job(f, orig_name or safe_name)
        return jsonify(job.to_status_dict()), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Lỗi khi đọc video: {e}"}), 500


@app.route("/api/jobs/<job_id>/start", methods=["POST"])
def api_start_job(job_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        job = manager.start_job(job_id, payload)
        return jsonify(job.to_status_dict()), 200
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/jobs/<job_id>/stop", methods=["POST"])
def api_stop_job(job_id: str):
    try:
        job = manager.stop_job(job_id)
        return jsonify(job.to_status_dict()), 200
    except KeyError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/jobs/<job_id>/status", methods=["GET"])
def api_job_status(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404
    return jsonify(job.to_status_dict()), 200


def _mjpeg_part(jpg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpg_bytes)).encode("ascii") + b"\r\n\r\n"
        + jpg_bytes
        + b"\r\n"
    )


@app.route("/api/jobs/<job_id>/preview", methods=["GET"])
def api_job_preview(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404
    jpg = manager.ensure_preview_jpeg(job)
    if jpg is None:
        return jsonify({"error": "Không đọc được khung hình preview từ video."}), 404
    return Response(
        jpg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/api/jobs/<job_id>/stream", methods=["GET"])
def api_job_stream(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404

    def generate():
        if job.status == "uploaded":
            jpg = manager.ensure_preview_jpeg(job)
            if jpg is not None:
                yield _mjpeg_part(jpg)
            yield b"--frame--\r\n"
            return

        last_seq = -1
        while True:
            with job.frame_cond:
                if job.frame_seq == last_seq and job.status in {"queued", "running"}:
                    job.frame_cond.wait(timeout=0.5)
                jpg = job.latest_jpeg
                cur_seq = job.frame_seq
                status = job.status

            if jpg is not None and cur_seq != last_seq:
                last_seq = cur_seq
                yield _mjpeg_part(jpg)
            elif status in {"completed", "stopped", "error"} and last_seq == -1:
                if jpg is None:
                    jpg = manager.ensure_preview_jpeg(job)
                last_seq = cur_seq
                if jpg is not None:
                    yield _mjpeg_part(jpg)

            if status in {"completed", "stopped", "error"} and cur_seq == last_seq:
                yield b"--frame--\r\n"
                break
            time.sleep(0.01)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/jobs/<job_id>/input", methods=["GET"])
def api_job_input(job_id: str):
    job = manager.get_job(job_id)
    if job is None or not job.input_path.exists():
        return jsonify({"error": "Không tìm thấy video gốc."}), 404
    mime, _ = mimetypes.guess_type(str(job.input_path))
    return send_file(
        str(job.input_path),
        mimetype=mime or "video/mp4",
        conditional=True,
    )


@app.route("/api/jobs/<job_id>/output", methods=["GET"])
def api_job_output(job_id: str):
    job = manager.get_job(job_id)
    if job is None or not job.output_path or not job.output_path.exists():
        return jsonify({"error": "Video kết quả chưa sẵn sàng."}), 404
    return send_file(
        str(job.output_path),
        mimetype=job.output_mime or "video/mp4",
        conditional=True,
    )


@app.route("/api/jobs/<job_id>/download", methods=["GET"])
def api_job_download(job_id: str):
    job = manager.get_job(job_id)
    if job is None or not job.output_path or not job.output_path.exists():
        return jsonify({"error": "Video kết quả chưa sẵn sàng."}), 404
    stem = Path(job.video_meta.get("filename", "drone_test")).stem
    dl_name = f"{stem}_{job.config.get('model_key', 'int8')}_{job.config.get('mode', 'track')}{job.output_path.suffix}"
    return send_file(
        str(job.output_path),
        mimetype=job.output_mime or "application/octet-stream",
        as_attachment=True,
        download_name=dl_name,
    )


@app.route("/api/jobs/<job_id>/tracks.json", methods=["GET"])
def api_job_tracks_json(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404
    p = job.job_dir / "tracks.json"
    if not p.exists():
        return jsonify({"error": "File tracks.json chưa sẵn sàng."}), 404
    as_dl = request.args.get("download") == "1"
    return send_file(
        str(p),
        mimetype="application/json",
        as_attachment=as_dl,
        download_name=f"{job_id}_tracks.json",
    )


@app.route("/api/jobs/<job_id>/tracks.csv", methods=["GET"])
def api_job_tracks_csv(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404
    p = job.job_dir / "tracks.csv"
    if not p.exists():
        return jsonify({"error": "File tracks.csv chưa sẵn sàng."}), 404
    return send_file(
        str(p),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{job_id}_tracks.csv",
    )


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def api_delete_job(job_id: str):
    ok = manager.delete_job(job_id)
    if not ok:
        return jsonify({"error": f"Không tìm thấy job {job_id}"}), 404
    return jsonify({"deleted": job_id}), 200


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26s-P2 INT8 Drone Detection & Tracking Web App")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    print("=" * 68, flush=True)
    print("  YOLO26s-P2 INT8 — DRONE DETECTION & TRACKING WEB TESTER", flush=True)
    print(f"  URL     : http://localhost:{args.port}", flush=True)
    print(f"  Models  : {[m['key'] for m in get_available_models()]}", flush=True)
    print(f"  Trackers: {[t['key'] for t in get_available_trackers()]}", flush=True)
    print(f"  Codec   : {detect_best_video_codec()}", flush=True)
    print("=" * 68, flush=True)

    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
