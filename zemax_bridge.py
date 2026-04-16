# -*- coding: utf-8 -*-
# 启动: C:\anaconda3\python.exe C:\zemax_bridge.py

import os, sys, io, math, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flask import Flask, request, jsonify, send_file

ZEMAX_DIR = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
sys.path.append(ZEMAX_DIR)

import clr
clr.AddReference(os.path.join(ZEMAX_DIR, "ZOSAPI"))
clr.AddReference(os.path.join(ZEMAX_DIR, "ZOSAPI_Interfaces"))
import ZOSAPI

app = Flask(__name__)
_conn = None
_app  = None
_sys  = None

def _init():
    global _conn, _app, _sys
    print("初始化 ZOS-API (CreateNewApplication)...")
    _conn = ZOSAPI.ZOSAPI_Connection()
    _app  = _conn.CreateNewApplication()
    _sys  = _app.PrimarySystem
    # 加载 CDGM 玻璃库
    try:
        cat = _sys.SystemData.MaterialCatalogs
        cat.AddCatalog("CDGM")
    except Exception as e:
        print(f"CDGM 加载失败（可忽略）: {e}")
    print("✓ ZOS-API 连接成功，启动 Flask 服务 port=5000")

def _get_system():
    return _sys

# ───────────────────────── /status ─────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"ok": True})

# ───────────────────────── /load_lens ──────────────────────
@app.route("/load_lens", methods=["POST"])
def load_lens():
    try:
        data  = request.json
        surfs = data.get("surfaces", [])
        fov   = float(data.get("fov",  30.0))
        fnum  = float(data.get("fnum",  2.8))
        sys_  = _get_system()

        sys_.New(False)

        # 孔径：像方F数
        sys_.SystemData.Aperture.ApertureType = \
            ZOSAPI.SystemData.ZemaxApertureType.ImageSpaceFNum
        sys_.SystemData.Aperture.ApertureValue = fnum

        # 视场
        fields = sys_.SystemData.Fields
        fields.DeleteAllFields()
        fields.AddField(0, 0, 1.0)
        half = fov / 2.0
        fields.AddField(0, half * 0.5, 1.0)
        fields.AddField(0, half,       1.0)

        # 波长
        wl = sys_.SystemData.Wavelengths
        wl.SelectWavelengthPreset(
            ZOSAPI.SystemData.WavelengthPreset.d_0p587)

        # 写面型（跳过超大厚度）
        lde = sys_.LDE
        while lde.NumberOfSurfaces < len(surfs) + 1:
            lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)

        for i, s in enumerate(surfs):
            surf = lde.GetSurfaceAt(i)
            r = s.get("radius", 0)
            t = s.get("thickness", 0)
            mat = s.get("material", "")
            if r and abs(r) < 1e8:
                surf.Radius = float(r)
            if t is not None and abs(t) < 1e6:
                surf.Thickness = float(t)
            if mat and mat.upper() not in ("AIR", ""):
                surf.Material = mat

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ───────────────────────── /layout ─────────────────────────
@app.route("/layout", methods=["POST"])
def layout():
    try:
        sys_ = _get_system()
        lde  = sys_.LDE
        n    = lde.NumberOfSurfaces

        # 收集面型
        surfaces = []
        z = 0.0
        for i in range(n):
            surf = lde.GetSurfaceAt(i)
            r = surf.Radius
            t = surf.Thickness
            sd = surf.SemiDiameter
            if abs(t) > 1e5:
                t = 0.0
            surfaces.append({"z": z, "r": r, "sd": sd, "t": t,
                              "mat": surf.Material})
            z += t

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        # 画镜片轮廓
        for s in surfaces:
            sd = s["sd"] if s["sd"] > 0 else 5.0
            ax.plot([s["z"], s["z"]], [-sd, sd],
                    color="#4a9eff", linewidth=1.5, alpha=0.8)

        # 追光线
        hy_vals = [-1.0, 0.0, 1.0]
        colors  = ["#ff6b6b", "#51cf66", "#339af0"]
        labels  = ["Field -1.0", "Field 0", "Field 1.0"]
        half_fov = sys_.SystemData.Fields.GetField(
            sys_.SystemData.Fields.NumberOfFields).Y

        for hy, col, lbl in zip(hy_vals, colors, labels):
            py_vals = [-0.8, -0.4, 0.0, 0.4, 0.8]
            for py in py_vals:
                rt = sys_.Tools.OpenBatchRayTrace()
                if rt is None:
                    continue
                max_rays = 1
                rays = rt.CreateNormUnpol(
                    max_rays,
                    ZOSAPI.Tools.RayTrace.RaysType.Real, False)
                rays.AddRay(1, hy, 0.0, py, 0.0,
                            ZOSAPI.Tools.RayTrace.OPDMode(0))
                rt.RunAndWaitForCompletion()
                rays.StartReadingResults()

                zpts, ypts = [], []
                for si in range(n):
                    result = rays.ReadNextResult()
                    # 兼容不同版本返回值数量
                    ok  = result[0]
                    ns  = result[1]
                    ys  = result[3]
                    zs  = result[4]
                    if ok and ns == 0:
                        zpts.append(zs)
                        ypts.append(ys)
                try:
                    rays.FinishReadingResults()
                except AttributeError:
                    pass
                rt.Close()

                if len(zpts) > 1:
                    ax.plot(zpts, ypts, color=col,
                            linewidth=0.6, alpha=0.55)

        for col, lbl in zip(colors, labels):
            ax.plot([], [], color=col, linewidth=1.5, label=lbl)
        ax.legend(facecolor="#1a1a2e", labelcolor="white",
                  fontsize=8, loc="upper left")
        ax.set_xlabel("Z (mm)", color="white")
        ax.set_ylabel("Y (mm)", color="white")
        ax.set_title("Lens Layout", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        # 裁剪 Y 轴
        all_sd = [s["sd"] for s in surfaces if 0 < s["sd"] < 1e4]
        if all_sd:
            ylim = max(all_sd) * 1.3
            ax.set_ylim(-ylim, ylim)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ───────────────────────── /zemax_optimize ─────────────────
@app.route("/zemax_optimize", methods=["POST"])
def zemax_optimize():
    try:
        cycles = int(request.json.get("cycles", 5))
        sys_   = _get_system()

        # 手动建 Merit Function（不用 Wizard 避免超时）
        mf = sys_.MFE
        n_ops = mf.NumberOfOperands
        if n_ops > 0:
            mf.RemoveOperandsAt(1, n_ops)

        num_fields = sys_.SystemData.Fields.NumberOfFields
        for fi in range(1, num_fields + 1):
            op = mf.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.TRAC)
            op.GetOperandCell(
                ZOSAPI.Editors.MFE.MeritColumn.Param1).IntegerValue = fi
            op.Weight = 1.0

        # Enum 显式转换（pythonnet 3 不支持隐式 int→Enum）
        AlgEnum = ZOSAPI.Tools.Optimization.OptimizationAlgorithm
        CycEnum = ZOSAPI.Tools.Optimization.OptimizationCycles

        opt = sys_.Tools.OpenLocalOptimization()
        if opt is None:
            return jsonify({"error": "OpenLocalOptimization 返回 None，系统无有效面型"}), 500

        opt.Algorithm = AlgEnum(0)                           # DampedLeastSquares
        opt.Cycles    = CycEnum(min(cycles, 5))              # 最多 5
        opt.RunAndWaitForCompletion()
        merit = opt.CurrentMeritFunction
        opt.Close()

        return jsonify({"merit": merit})
    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500

# ───────────────────────── main ────────────────────────────
if __name__ == "__main__":
    _init()
    app.run(host="0.0.0.0", port=5000, debug=False)
