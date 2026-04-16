"""
Windows 端 Zemax 桥接服务
环境: Windows + Ansys Zemax OpticStudio 2024 R1.00 + pythonnet 3.x
启动: C:\anaconda3\python.exe C:\zemax_bridge.py
依赖: pip install flask matplotlib
"""

import os, sys, tempfile
import clr

# ── ZOS-API 初始化 ────────────────────────────────────────────────────────────
ZEMAX_DIR = r"D:\Program Files\Ansys Zemax OpticStudio 2024 R1.00"
sys.path.append(ZEMAX_DIR)
clr.AddReference(os.path.join(ZEMAX_DIR, "ZOSAPI"))
clr.AddReference(os.path.join(ZEMAX_DIR, "ZOSAPI_Interfaces"))
import ZOSAPI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file

app   = Flask(__name__)
_conn = None   # 必须全局持有，否则 GC 回收后 Zemax 自动关闭
_zos  = None
_sys  = None


def _get_system():
    global _conn, _zos, _sys
    if _zos is None:
        _conn = ZOSAPI.ZOSAPI_Connection()
        _zos  = _conn.CreateNewApplication()
        _sys  = _zos.PrimarySystem
        # 加载 CDGM 玻璃库（只需一次）
        try:
            _sys.SystemData.MaterialCatalogs.AddCatalog("CDGM")
        except Exception:
            pass
    return _sys


# ── /status ───────────────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    try:
        _get_system()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── /load_lens ────────────────────────────────────────────────────────────────
@app.route("/load_lens", methods=["POST"])
def load_lens():
    """
    接收面型数据，写入 Zemax LDE，并设置孔径和视场。
    Body JSON: { "surfaces": [...], "fov": float, "fnum": float }
    每个 surface: { surface_num, radius, thickness, material, semi_diameter }
    """
    data  = request.json
    surfs = data.get("surfaces", [])
    fov   = data.get("fov")    # 全视角（度），半角 = fov/2
    fnum  = data.get("fnum")

    try:
        sys_ = _get_system()
        lde  = sys_.LDE

        # 确保面数足够
        needed = max(int(s.get("surface_num", 0)) for s in surfs) + 2
        while lde.NumberOfSurfaces < needed:
            lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)

        for s in surfs:
            idx = int(s.get("surface_num", 0))
            if idx < 0 or idx >= lde.NumberOfSurfaces:
                continue
            surf = lde.GetSurfaceAt(idx)
            r   = s.get("radius")
            t   = s.get("thickness")
            mat = s.get("material", "")
            sd  = s.get("semi_diameter")
            if r is not None and abs(r) < 1e8:
                surf.Radius = float(r)
            if t is not None:
                surf.Thickness = float(t)
            if mat and str(mat).upper() not in ("AIR", ""):
                surf.Material = str(mat)
            if sd is not None and float(sd) > 0:
                surf.SemiDiameter = float(sd)

        # ── 孔径：设为像方 F 数 ──────────────────────────────────────────────
        if fnum is not None:
            ap = sys_.SystemData.Aperture
            ap.ApertureType  = ZOSAPI.SystemData.ZemaxApertureType.ImageSpaceFNum
            ap.ApertureValue = float(fnum)

        # ── 视场：角度类型，3 个视场点（0 / 0.7 / 1.0 归一化半角）────────────
        if fov is not None:
            half = float(fov) / 2.0
            sfd  = sys_.SystemData.Fields
            sfd.SetFieldType(ZOSAPI.SystemData.FieldType.Angle)
            while sfd.NumberOfFields > 1:
                sfd.RemoveField(sfd.NumberOfFields)
            sfd.GetField(1).Y = 0.0          # Field 1: 轴上
            sfd.AddField(0, half * 0.7, 1.0) # Field 2: 0.7 视场
            sfd.AddField(0, half * 1.0, 1.0) # Field 3: 全视场

        # ── 建默认 Merit Function（RMS 弥散斑）────────────────────────────────
        try:
            mfe = sys_.MFE
            # 清空旧操作数
            while mfe.NumberOfOperands > 0:
                mfe.RemoveOperandAt(1)
            # 用内置向导生成默认 Merit Function（RMS 波前，所有视场/波长）
            wizard = mfe.SEQOptimizationWizard
            wizard.Type     = 0   # RMS
            wizard.Data     = 0   # 波前
            wizard.Ring     = 3   # 3 环
            wizard.Arm      = 6   # 6 臂
            wizard.IsUsed   = True
            wizard.CommonSettings.IsMeritDefault = True
            wizard.Apply()
        except Exception:
            pass  # 向导失败时跳过，optimize 时再报错

        return jsonify({"ok": True, "n_surfaces": lde.NumberOfSurfaces})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /layout ───────────────────────────────────────────────────────────────────
@app.route("/layout", methods=["POST"])
def layout():
    """
    子午面光线追迹 → matplotlib 2D Layout PNG
    pythonnet 3.x: result = (ok, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, intensity)
    """
    try:
        sys_   = _get_system()
        n_surf = sys_.LDE.NumberOfSurfaces

        # 全局 Z：累加厚度，跳过占位符（像面距通常 >1e6）
        z_globals = [0.0]
        for i in range(n_surf - 1):
            t = float(sys_.LDE.GetSurfaceAt(i).Thickness)
            if abs(t) > 1e6:
                t = 0.0
            z_globals.append(z_globals[-1] + t)

        rt  = sys_.Tools.OpenBatchRayTrace()
        if rt is None:
            return jsonify({"error": "OpenBatchRayTrace 返回 None，请重启桥接服务"}), 500
        fig, ax = plt.subplots(figsize=(14, 7))

        colors  = ["steelblue", "forestgreen", "crimson"]
        labels  = ["Field -1.0", "Field 0", "Field 1.0"]
        hy_vals = [-1.0, 0.0, 1.0]

        for field_idx, (hy, color, label) in enumerate(zip(hy_vals, colors, labels)):
            for py in [-0.8, -0.4, 0.0, 0.4, 0.8]:
                zz, yy = [0.0], [0.0]
                ok_all = True
                for s in range(1, n_surf):
                    try:
                        result = rt.SingleRayNormUnpol(
                            ZOSAPI.Tools.RayTrace.RaysType.Real,
                            s,      # toSurface
                            1,      # waveNumber（第1波长）
                            0.0,    # Hx
                            hy,     # Hy（归一化场高）
                            0.0,    # Px
                            py,     # Py（归一化光瞳）
                            False,
                            0, 0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0
                        )
                        y_val = float(result[4])
                        if abs(y_val) > 1e4:   # 光线发散到极端值，停止追迹
                            ok_all = False
                            break
                        zz.append(z_globals[s])
                        yy.append(y_val)
                    except Exception:
                        ok_all = False
                        break
                if ok_all and len(zz) > 1:
                    ax.plot(zz, yy, color=color, linewidth=0.8, alpha=0.6,
                            label=label if py == 0.0 else None)

        rt.Close()

        # 镜片轮廓线（SemiDiameter 超过 200mm 的跳过，是 Zemax 自动计算的异常值）
        for i in range(n_surf):
            sd = float(sys_.LDE.GetSurfaceAt(i).SemiDiameter or 0)
            if 0 < sd < 200:
                ax.plot([z_globals[i], z_globals[i]], [-sd, sd],
                        color="gray", linewidth=1.2, alpha=0.8)

        ax.set_xlabel("Z (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("2D Lens Layout — Y-Z Meridional")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        handles = [plt.Line2D([0], [0], color=colors[i], label=labels[i])
                   for i in range(3)]
        ax.legend(handles=handles, loc="upper left")
        plt.tight_layout()

        tmp_path = os.path.join(tempfile.gettempdir(), "zx_layout.png")
        plt.savefig(tmp_path, dpi=150)
        plt.close(fig)
        return send_file(tmp_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /zemax_optimize ───────────────────────────────────────────────────────────
@app.route("/zemax_optimize", methods=["POST"])
def zemax_optimize():
    """DLS 局部优化。Body: { "cycles": int }  返回: { "merit": float }"""
    cycles = max(1, min(20, int(request.json.get("cycles", 5))))
    try:
        sys_ = _get_system()

        # 确保有 Merit Function（默认 RMS 波前）
        mfe = sys_.MFE
        if mfe.NumberOfOperands == 0:
            sys_.Tools.ConvertToSequential()   # 确保序列模式
            mfe.AddOperand()                   # 加一个默认操作数触发计算

        opt = sys_.Tools.OpenLocalOptimization()
        if opt is None:
            return jsonify({"error": "OpenLocalOptimization 返回 None，Merit Function 可能为空，请先调用 load_lens 写入面型"}), 500

        # pythonnet 3: Enum 不能隐式转换，需显式指定
        opt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm(0)  # 0 = DampedLeastSquares
        opt.Cycles    = cycles
        opt.RunAndWaitForCompletion()
        merit = float(opt.CurrentMeritFunction)
        opt.Close()
        return jsonify({"merit": merit})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("初始化 ZOS-API (CreateNewApplication)...")
    _get_system()
    print("ZOS-API 连接成功，启动 Flask 服务 port=5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
