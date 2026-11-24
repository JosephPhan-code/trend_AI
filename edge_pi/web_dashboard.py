"""
==============================================================================
Module: web_dashboard.py
Mô tả : Giao diện Web Dashboard Flask (T-Web) & Hiển thị màn hình HDMI cục bộ (T-HDMI).
==============================================================================
"""

import json
from queue import Empty
import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from config import (
    LOCAL_DISPLAY_ENABLED,
    LOCAL_FULLSCREEN,
    PID_FILE,
    STATE,
    STOP_EVENT,
    WEB_HOST,
    WEB_PORT,
    q_local,
    q_stream,
    validate_pid,
)

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AEGIS — Drone Tracker</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--g:#00ffcc;--r:#ff4444;--b:#00aaff;--bg:#060a12;--card:rgba(0,255,204,.04);--brd:rgba(0,255,204,.15);--t:#8a9ba8}
body{background:var(--bg);color:var(--g);font-family:'Courier New',monospace;min-height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;justify-content:space-between;align-items:center;padding:10px 20px;background:rgba(0,255,204,.04);border-bottom:1px solid var(--brd);flex-shrink:0}
.logo{font-size:17px;font-weight:bold;letter-spacing:4px}.logo em{color:var(--r);font-style:normal}
.hr{display:flex;align-items:center;gap:14px;font-size:12px;color:var(--t)}
.dot{width:8px;height:8px;border-radius:50%;background:#00ff88;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.body{display:grid;grid-template-columns:1fr 300px;gap:10px;padding:10px 20px;flex:1;min-height:0;overflow:hidden}
.vp{position:relative;background:#000;border:1px solid var(--brd);border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.vp img{width:100%;height:100%;object-fit:contain}
.co{position:absolute;width:16px;height:16px;border-color:var(--g);border-style:solid}
.tl{top:8px;left:8px;border-width:2px 0 0 2px}.tr{top:8px;right:8px;border-width:2px 2px 0 0}
.bl{bottom:8px;left:8px;border-width:0 0 2px 2px}.br{bottom:8px;right:8px;border-width:0 2px 2px 0}
.lb{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:rgba(255,68,68,.12);border:1px solid var(--r);color:var(--r);padding:2px 12px;border-radius:20px;font-size:11px;letter-spacing:3px;display:flex;align-items:center;gap:5px}
.sb{display:flex;flex-direction:column;gap:8px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--brd) transparent}
.card{background:var(--card);border:1px solid var(--brd);border-radius:8px;padding:12px}
.ct{font-size:10px;letter-spacing:2px;color:var(--t);text-transform:uppercase;margin-bottom:8px}
.bb{font-size:10px;color:var(--b);letter-spacing:1px;text-align:center;padding:3px;border:1px solid rgba(0,170,255,.3);border-radius:4px;margin-bottom:6px}
.fb{font-size:42px;font-weight:bold;line-height:1}.fu{font-size:12px;color:var(--t)}
.bg{background:rgba(0,255,204,.08);border-radius:3px;height:5px;overflow:hidden;margin-top:6px}
.ba{height:100%;border-radius:3px;transition:width .3s}.fb2{background:linear-gradient(90deg,#00cc99,#00ffcc)}.cb{background:linear-gradient(90deg,#ff9d00,var(--r))}
.dc{font-size:52px;font-weight:bold;text-align:center;line-height:1;transition:.3s}
.dc.z{color:#00ff88;text-shadow:0 0 20px rgba(0,255,136,.4)}.dc.h{color:var(--r);text-shadow:0 0 20px rgba(255,68,68,.4)}
.dl{text-align:center;font-size:10px;color:var(--t);margin-top:4px;letter-spacing:2px}
.sr{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(0,255,204,.07)}
.sr:last-child{border:none;padding-bottom:0}
.sl{font-size:11px;color:var(--t)}.sv{font-size:12px;font-weight:bold}
.gd{color:#00ff88}.wn{color:#ff9d00}
.pa{font-size:11px;color:var(--b);letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid rgba(0,170,255,.2);padding-bottom:4px}
.pr{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.pl{font-size:10px;color:var(--t);width:24px;flex-shrink:0}
.ps{flex:1;-webkit-appearance:none;height:4px;border-radius:2px;background:rgba(0,255,204,.15);outline:none;cursor:pointer}
.ps::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--g);cursor:pointer}
.pv{font-size:11px;color:var(--g);width:52px;text-align:right;flex-shrink:0}
.btn{width:100%;padding:7px;border:1px solid var(--g);border-radius:5px;background:rgba(0,255,204,.08);color:var(--g);font-family:inherit;font-size:12px;letter-spacing:2px;cursor:pointer;transition:.2s}
.btn:hover{background:rgba(0,255,204,.18)}
.br2{border-color:var(--r);color:var(--r);background:rgba(255,68,68,.06)}.br2:hover{background:rgba(255,68,68,.15)}
.brow{display:flex;gap:6px}
.pt{font-size:11px;text-align:center;color:#00ff88;margin-top:5px;min-height:16px;opacity:0;transition:opacity .3s}
.ab{border-radius:6px;padding:8px;font-size:11px;text-align:center;letter-spacing:1px;font-weight:bold;transition:.3s}
.ab.cl{border:1px solid rgba(0,255,136,.3);background:rgba(0,255,136,.04);color:#00ff88}
.ab.dn{border:1px solid rgba(255,68,68,.4);background:rgba(255,68,68,.07);color:#ff6666;animation:pulse .8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
</style>
</head>
<body>
<header>
  <div class="logo">AEGIS <em>▲</em> TACTICAL</div>
  <div class="hr"><div class="dot"></div><span id="bhdr">INIT...</span><span id="clk">--:--:--</span></div>
</header>
<div class="body">
  <div class="vp">
    <div class="co tl"></div><div class="co tr"></div><div class="co bl"></div><div class="co br"></div>
    <div class="lb"><div class="dot"></div>LIVE</div>
    <img src="/video_feed" alt="Live">
  </div>
  <div class="sb">
    <div class="card">
      <div class="ct">Processing Speed</div>
      <div class="bb" id="bbd">INIT...</div>
      <div><span class="fb" id="fv">0</span><span class="fu"> FPS</span></div>
      <div class="bg"><div class="ba fb2" id="fb" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="ct">Targets In Frame</div>
      <div class="dc z" id="dc">0</div>
      <div class="dl">DRONE(S) DETECTED</div>
    </div>
    <div class="card">
      <div class="ct">System Stats</div>
      <div class="sr"><span class="sl">Confidence</span><span class="sv" id="cv2">0%</span></div>
      <div class="bg" style="margin-bottom:6px"><div class="ba cb" id="cb" style="width:0%"></div></div>
      <div class="sr"><span class="sl">Servo Pan</span><span class="sv gd" id="pv">0.0°</span></div>
      <div class="sr"><span class="sl">Servo Tilt</span><span class="sv gd" id="tv">0.0°</span></div>
      <div class="sr"><span class="sl">Error X</span><span class="sv wn" id="ex">0px</span></div>
      <div class="sr"><span class="sl">Error Y</span><span class="sv wn" id="ey">0px</span></div>
      <div class="sr"><span class="sl">Track ID</span><span class="sv gd" id="tid">-</span></div>
      <div class="sr"><span class="sl">Uptime</span><span class="sv" id="up">00:00:00</span></div>
      <div class="sr"><span class="sl">Total Detects</span><span class="sv" id="td">0</span></div>
    </div>
    <div class="card">
      <div class="ct">PID Tuner (Live)</div>
      <div class="pa">PAN (Truc ngang)</div>
      <div class="pr"><span class="pl">Kp</span><input type="range" class="ps" id="pk" min="0" max="0.2" step="0.001" value="{{pid.pan_kp}}" oninput="ul('pkl',this.value)"><span class="pv" id="pkl">{{pid.pan_kp}}</span></div>
      <div class="pr"><span class="pl">Ki</span><input type="range" class="ps" id="pi2" min="0" max="0.01" step="0.0001" value="{{pid.pan_ki}}" oninput="ul('pil',this.value)"><span class="pv" id="pil">{{pid.pan_ki}}</span></div>
      <div class="pr"><span class="pl">Kd</span><input type="range" class="ps" id="pd" min="0" max="0.1" step="0.001" value="{{pid.pan_kd}}" oninput="ul('pdl',this.value)"><span class="pv" id="pdl">{{pid.pan_kd}}</span></div>
      <div class="pa">TILT (Truc doc)</div>
      <div class="pr"><span class="pl">Kp</span><input type="range" class="ps" id="tk" min="0" max="0.2" step="0.001" value="{{pid.tilt_kp}}" oninput="ul('tkl',this.value)"><span class="pv" id="tkl">{{pid.tilt_kp}}</span></div>
      <div class="pr"><span class="pl">Ki</span><input type="range" class="ps" id="ti" min="0" max="0.01" step="0.0001" value="{{pid.tilt_ki}}" oninput="ul('til',this.value)"><span class="pv" id="til">{{pid.tilt_ki}}</span></div>
      <div class="pr"><span class="pl">Kd</span><input type="range" class="ps" id="td2" min="0" max="0.1" step="0.001" value="{{pid.tilt_kd}}" oninput="ul('tdl',this.value)"><span class="pv" id="tdl">{{pid.tilt_kd}}</span></div>
      <div class="pa">Deadband (Vung chet)</div>
      <div class="pr"><span class="pl">px</span><input type="range" class="ps" id="db" min="0" max="50" step="1" value="{{pid.deadband}}" oninput="ul('dbl',this.value)"><span class="pv" id="dbl">{{pid.deadband}}</span></div>
      <div class="brow" style="margin-top:8px">
        <button class="btn" onclick="applyPID()">AP DUNG</button>
        <button class="btn br2" onclick="resetPID()">MAC DINH</button>
      </div>
      <button class="btn br2" style="margin-top:6px;width:100%;font-weight:bold;letter-spacing:1px;background:rgba(255,68,68,0.15);" onclick="shutdownSystem()">⏹ TAT HE THONG</button>
      <div class="pt" id="pt">Da cap nhat PID!</div>
    </div>
    <div class="ab cl" id="ab">AIRSPACE CLEAR</div>
  </div>
</div>
<script>
setInterval(()=>{document.getElementById('clk').textContent=new Date().toLocaleTimeString('vi-VN',{hour12:false});},1000);
function ul(id,v){document.getElementById(id).textContent=parseFloat(v).toFixed(4);}
async function fs(){
  try{
    const d=await(await fetch('/api/stats')).json();
    const f=Math.round(d.ai_fps);
    document.getElementById('fv').textContent=f;
    document.getElementById('fb').style.width=Math.min(f/60*100,100)+'%';
    document.getElementById('bbd').textContent=d.backend;
    document.getElementById('bhdr').textContent=d.backend;
    const n=d.drone_count;
    const el=document.getElementById('dc');
    el.textContent=n;el.className='dc '+(n===0?'z':'h');
    document.getElementById('cv2').textContent=d.max_conf+'%';
    document.getElementById('cb').style.width=Math.min(d.max_conf,100)+'%';
    document.getElementById('pv').textContent=d.pan_angle+'°';
    document.getElementById('tv').textContent=d.tilt_angle+'°';
    document.getElementById('ex').textContent=d.error_x+'px';
    document.getElementById('ey').textContent=d.error_y+'px';
    document.getElementById('tid').textContent=d.track_id>0 ? ('#'+d.track_id) : '-';
    document.getElementById('td').textContent=d.total_detect;
    const s=Math.floor(d.uptime);
    document.getElementById('up').textContent=
      [Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(v=>String(v).padStart(2,'0')).join(':');
    const ab=document.getElementById('ab');
    if (d.tracker_state === 'LOCKED') {
        ab.className = 'ab dn';
        ab.style.color = '#ff6666';
        ab.style.borderColor = 'rgba(255,68,68,0.4)';
        ab.style.background = 'rgba(255,68,68,0.07)';
        ab.textContent = 'LOCKED: TARGET ACQUIRED';
    } else {
        ab.className = 'ab cl';
        ab.style.color = '#00ff88';
        ab.style.borderColor = 'rgba(0,255,136,0.3)';
        ab.style.background = 'rgba(0,255,136,0.04)';
        ab.textContent = 'IDLE: AIRSPACE CLEAR';
    }
  }catch(e){}
}
setInterval(fs,500);fs();
async function applyPID(){
  const p={
    pan_kp:+document.getElementById('pk').value,
    pan_ki:+document.getElementById('pi2').value,
    pan_kd:+document.getElementById('pd').value,
    tilt_kp:+document.getElementById('tk').value,
    tilt_ki:+document.getElementById('ti').value,
    tilt_kd:+document.getElementById('td2').value,
    deadband:+document.getElementById('db').value,
  };
  const r=await fetch('/api/set_pid',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
  if(r.ok){const t=document.getElementById('pt');t.style.opacity='1';setTimeout(()=>t.style.opacity='0',2500);}
}
const D={'pk':{{pid.pan_kp}},'pi2':{{pid.pan_ki}},'pd':{{pid.pan_kd}},'tk':{{pid.tilt_kp}},'ti':{{pid.tilt_ki}},'td2':{{pid.tilt_kd}},'db':{{pid.deadband}}};
const L={'pk':'pkl','pi2':'pil','pd':'pdl','tk':'tkl','ti':'til','td2':'tdl','db':'dbl'};
function resetPID(){for(const[id,v]of Object.entries(D)){document.getElementById(id).value=v;ul(L[id],v);}applyPID();}
async function shutdownSystem(){
  if(confirm('Ban co chac chan muon tat he thong Drone Tracker?')){
    try{
      await fetch('/api/shutdown',{method:'POST'});
      alert('He thong dang tat an toan... Vui long doi!');
    }catch(e){}
  }
}
</script>
</body></html>"""


def _gen_stream():
    STATE.stream_viewers += 1
    try:
        while not STOP_EVENT.is_set():
            try:
                data = q_stream.get(timeout=1.0)
            except Empty:
                continue
            ret, buf = cv2.imencode(".jpg", data["annotated"], [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ret:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        STATE.stream_viewers = max(0, STATE.stream_viewers - 1)


@app.route("/")
def index():
    return render_template_string(HTML, pid=STATE.pid_params)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    return jsonify(STATE.get_stats())


@app.route("/api/set_pid", methods=["POST"])
def api_set_pid():
    body = request.get_json(silent=True) or {}
    try:
        params = validate_pid(body)
        if not params:
            return jsonify({"error": "no valid params"}), 400
        STATE.set_pid(params)
    except (TypeError, ValueError, OverflowError) as e:
        return jsonify({"error": str(e)}), 400

    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE.pid_params, f, indent=4)
        print("[CONFIG] Da luu cau hinh PID vao file.")
    except Exception as e:
        print(f"[CONFIG] Loi luu file PID: {e}")

    return jsonify({"ok": True})


@app.route("/api/shutdown", methods=["POST", "GET"])
def api_shutdown():
    print("\n[WEB] Nhan lenh tat he thong tu Web Dashboard...")
    STOP_EVENT.set()
    return jsonify({"ok": True, "message": "He thong dang tat an toan..."})


class WebDashboardServer:
    """Đóng gói OOP cho Flask Web Dashboard server."""

    def __init__(self, host=WEB_HOST, port=WEB_PORT):
        self.host = host
        self.port = port

    def run(self):
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        print(f"\n[WEB]  Dashboard: http://localhost:{self.port}")
        print(f"[WEB]  Tu may tinh: http://<IP_PI>:{self.port}\n")
        app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)


class LocalDisplayWorker:
    """Đóng gói OOP cho cửa sổ hiển thị HDMI trực tiếp trên Pi."""

    def __init__(self, local_queue=None, stop_event=None):
        self.local_queue = local_queue or q_local
        self.stop_event = stop_event or STOP_EVENT

    def run(self):
        if not LOCAL_DISPLAY_ENABLED:
            print("[HDMI] Local display tat (set DRONE_TRACKER_HDMI=1 de bat).")
            return
        window = "AEGIS - Drone Tracker"

        def _on_mouse(event, x, y, flags, param):
            if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
                action = "CHUOT PHAI" if event == cv2.EVENT_RBUTTONDOWN else "DOUBLE CLICK"
                print(f"\n[HDMI] Phat hien {action} tai ({x}, {y}) -> DANG TAT HE THONG AN TOAN...")
                self.stop_event.set()

        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if LOCAL_FULLSCREEN:
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(window, _on_mouse)
            print("[HDMI] Local display bat. CLICK CHUOT PHAI hoac nhan Q/Esc de thoat.")
            while not self.stop_event.is_set():
                try:
                    data = self.local_queue.get(timeout=0.5)
                except Empty:
                    cv2.waitKey(1)
                    continue
                cv2.imshow(window, data["annotated"])
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("\n[HDMI] Nhan phim thoat -> DANG TAT HE THONG...")
                    self.stop_event.set()
                    break
        except Exception as e:
            print(f"[HDMI] Khong mo duoc cua so local: {e}")
        finally:
            self.stop_event.set()
            try:
                cv2.destroyWindow(window)
            except Exception:
                pass


def thread_web():
    WebDashboardServer().run()


def thread_local_display():
    LocalDisplayWorker().run()
