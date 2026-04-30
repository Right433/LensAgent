# -*- coding: utf-8 -*-
# 启动: C:\anaconda3\python.exe C:\zemax_bridge.py
#
# 相对原版改动（修步骤5精优的几个致命问题）:
#   P0  /zemax_optimize: 优化前把所有玻璃面的 radius 标为变量（原版空跑）
#   P1  Cycles 枚举正确映射（原版把 int 当 enum，cycles=5 被当成 Infinite）
#   P2  Merit function: 加 EFFL 目标约束 + RMS spot size 操作数（按视场）
#   P3  新增 /metrics endpoint: 回传 effl/fnum/totr/rms_per_field/distortion
#   P4  波长改 F-d-C 三波长，色差相关 skill 才有意义
#   P7  /load_lens 接收可选 stop_surface
#   P8  ★ 致命修复：sys_.New(False) 会重置 MaterialCatalogs,必须在 New() 后
#       重新 AddCatalog("CDGM"),否则 H-ZF4A/H-ZLAF55D 等国产玻璃
#       全部被当成 n=1 空气,EFFL=1e10,优化完全不动。(症状:TOTR 正常但
#       EFL=∞,RSCE 三视场完全相等于 0.7071)
#   P8b 加"玻璃真被识别了吗"的边缘光线探针 —— 轴上追迹无法区分有无折射
#   P9  Merit 权重重配: RSCE/EFFL/WFNO 从 1/10/5 改为 4/3/2.
#       旧权重让约束压碎像质(F/1.2 优化后 RMS 反而从 0.031→0.224 变差).
#       新权重让像质主导,EFFL/WFNO 轻约束即可(agent 已先用 STOP SD 对齐 F#)
#   P10 /zemax_optimize 返回 surfaces_after (优化后的完整面型),
#       agent 能回写本地 lens.surfaces 避免用老 radius 误判 RMS
#   P11 边缘光探针从 BatchRayTrace 改为 MFE REAY operand,免受版本字段差异影响
#   P13 ★ 玻璃厚度从优化变量中移除（只有空气间隔和曲率半径是变量）
#       彻底杜绝沙漏形镜片：CTGT/MNCG 是软约束，DLS 可绕开；
#       把玻璃厚度固定才是硬保证。CTGT target 改为 max(当前厚度×0.9, 1mm)。
#   P14 FCUR weight 15→8，RSCE 边缘视场梯度权重 (4/5/6)，MXSD 净口径上限约束
#   P12 ★ _init() 重构: 加 License 诊断 + Standalone 失败自动回退 ConnectAsExtension
#       + PrimarySystem=None 时用 CreateNewSystem(Sequential) 兜底, 修复
#       "'NoneType' object has no attribute 'SystemData'" 那一类由授权或 GUI
#       冲突导致的假性"连上但拿不到系统"的场景.

import os, sys, io, math, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _ensure_cdgm_catalog(sys_, label="ensure_cdgm"):
    """★ P8 核心修复：
    sys_.New(False) 会把 MaterialCatalogs 重置为默认（仅 Schott）,之前挂的 CDGM
    会被清掉。因此每次 load_lens / 新建系统后都必须重新 AddCatalog("CDGM"),
    否则 surf.Material = "H-ZF4A" 字符串能写入但查不到折射率,Zemax 静默当 n=1,
    EFFL 返回 1e10 哨兵值,优化器永远找不到下降方向。

    2024 R1 实测：MaterialCatalogs 对象是 ISDMaterialCatalogData,
    没有 NumberOfCatalogs/GetCatalog 这种 collection-style API,
    只能 AddCatalog 后靠后续 surf.Material 行为判断成功与否。
    所以这里只做 AddCatalog 幂等调用,不做回读 —— 回读一定失败且噪音大。

    真正的功能验证在 load_lens 的边缘光探针那里: 如果 EFFL 不是哨兵 1e10,
    说明玻璃 catalog 已生效。

    返回 (ok:bool, note:str) — ok 为 False 只代表 AddCatalog 本身抛了非
    "already exists" 的异常;
    """
    try:
        cat = sys_.SystemData.MaterialCatalogs
    except AttributeError:
        # 极老/极新版本可能叫 MaterialsCatalog
        try:
            cat = sys_.SystemData.MaterialsCatalog
        except Exception as e:
            print(f"[DIAG {label}] ❌ SystemData 上找不到 MaterialCatalogs: {e}")
            return False, f"attr not found: {e}"

    try:
        cat.AddCatalog("CDGM")
        return True, "AddCatalog('CDGM') 调用成功"
    except Exception as e:
        # 已存在 / 重复添加,大多数版本这是正常的
        msg = str(e).lower()
        if "exist" in msg or "already" in msg:
            return True, f"已存在: {e}"
        print(f"[DIAG {label}] ⚠ AddCatalog('CDGM') 抛异常: {e}")
        return False, str(e)


def _init():
    """初始化 ZOS-API。
    连接策略（按顺序尝试,第一个成功的为准）:
      1) CreateNewApplication (Standalone) —— 需要 Premium 授权
      2) ConnectAsExtension(0) —— 需要 GUI 打开且启用 Interactive Extension
      3) 兜底: 若 PrimarySystem=None 但 License 有效, 用 CreateNewSystem(Sequential)
    每一步都打印诊断信息, 让根因暴露在日志里而不是被 NoneType 掩盖。
    """
    global _conn, _app, _sys
    _conn = ZOSAPI.ZOSAPI_Connection()
    _app  = None
    _mode = None

    # ── 尝试 1: Standalone (CreateNewApplication) ────────────────
    print("初始化 ZOS-API [尝试 1/2]: CreateNewApplication (Standalone)...")
    try:
        _app = _conn.CreateNewApplication()
    except Exception as e:
        print(f"  ❌ CreateNewApplication 抛异常: {e}")
        _app = None

    if _app is not None:
        try:
            lic_status = _app.LicenseStatus
            lic_valid  = bool(_app.IsValidLicenseForAPI)
            print(f"  [DIAG] LicenseStatus        = {lic_status}")
            print(f"  [DIAG] IsValidLicenseForAPI = {lic_valid}")
            if lic_valid:
                _mode = "Standalone"
            else:
                print("  ⚠ License 不允许 Standalone ZOS-API, 尝试 Extension 模式...")
                try:
                    _app.CloseApplication()
                except Exception:
                    pass
                _app = None
        except Exception as e:
            print(f"  ⚠ 读取 License 状态失败: {e}, 尝试 Extension 模式...")
            _app = None

    # ── 尝试 2: Interactive Extension ───────────────────────────
    if _app is None:
        print("初始化 ZOS-API [尝试 2/2]: ConnectAsExtension(0) (需要 GUI + "
              "Programming→Interactive Extension 已启用)...")
        try:
            _app = _conn.ConnectAsExtension(0)
            if _app is not None:
                lic_valid = bool(_app.IsValidLicenseForAPI)
                print(f"  [DIAG] IsValidLicenseForAPI = {lic_valid}")
                if lic_valid:
                    _mode = "Extension"
                else:
                    print("  ❌ Extension 模式 License 仍无效")
                    _app = None
        except Exception as e:
            print(f"  ❌ ConnectAsExtension 失败: {e}")
            _app = None

    if _app is None:
        raise RuntimeError(
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "两种连接方式都失败,请按以下步骤排查:\n"
            "  ① 检查架构: python -c \"import platform; print(platform.architecture())\"\n"
            "     必须是 64bit (与 OpticStudio 一致)\n"
            "  ② 若要用 Standalone: 需 Premium 级别授权 (联系 Ansys 确认)\n"
            "  ③ 若要用 Extension: \n"
            "      - 手动打开 OpticStudio GUI\n"
            "      - 顶部 Programming 选项卡 → 点击 Interactive Extension\n"
            "      - 按钮变成 Terminate Interactive Extension 即为启用\n"
            "      - 然后重新运行本脚本\n"
            "  ④ 确认 ZEMAX_DIR 路径正确: " + ZEMAX_DIR + "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    print(f"  ✓ 连接模式: {_mode}")

    # ── 获取 PrimarySystem, None 时兜底用 CreateNewSystem ────────
    _sys = _app.PrimarySystem
    if _sys is None:
        print("[DIAG init] PrimarySystem=None, 尝试 CreateNewSystem(Sequential)...")
        try:
            _sys = _app.CreateNewSystem(ZOSAPI.SystemType.Sequential)
        except Exception as e:
            print(f"[DIAG init] CreateNewSystem 失败: {e}")
    if _sys is None:
        raise RuntimeError(
            "PrimarySystem 为 None 且 CreateNewSystem 也失败,无法继续。\n"
            "这通常意味着 License 虽然通过 IsValidLicenseForAPI 检查,\n"
            "但实际授权不包含当前操作所需的模块。"
        )
    print(f"[DIAG init] ✓ PrimarySystem 获取成功 (Mode: {_mode})")

    ok, note = _ensure_cdgm_catalog(_sys, label="init")
    if ok:
        print(f"[DIAG] ✓ CDGM catalog 加载成功 ({note})")
    else:
        print(f"[DIAG] ❌ CDGM 加载失败（国产玻璃会无法识别）: {note}")
    print("✓ ZOS-API 连接成功，启动 Flask 服务 port=5000")


def _get_system():
    return _sys


# ═════════════════════════════════════════════════════════════
#   共用：读系统真值指标（供 /metrics 和 /zemax_optimize 复用）
# ═════════════════════════════════════════════════════════════
def _read_system_metrics(sys_):
    """返回 {effl, fnum, totr, rms_per_field_mm, distortion_per_field_pct}

    用 MFE.GetOperandValue 单次求值，不插 MFE 也不依赖 op.Value 的状态同步
    （interactive / standalone 模式下 AddOperand + CalculateMeritFunction 后
    op.Value 有时不刷新，是已知坑）。
    视场区分用归一化 Hy 坐标，不是 Param1（Param1 是 Sampling，塞 field_number 是错的）。
    """
    OpType = ZOSAPI.Editors.MFE.MeritOperandType
    mfe = sys_.MFE
    out = {}

    def _gov(op_type, *args):
        """薄封装：GetOperandValue(Type, P1..P8)，参数不够自动补 0。"""
        padded = list(args) + [0] * (8 - len(args))
        try:
            v = mfe.GetOperandValue(op_type, *padded)
            return round(float(v), 6)
        except Exception as e:
            print(f"[DIAG metrics] GetOperandValue({op_type}) 失败: {e}")
            return None

    # 全系统量：所有参数传 0
    out["effl"] = _gov(OpType.EFFL)
    out["fnum"] = _gov(OpType.WFNO)   # image-space working F#
    out["totr"] = _gov(OpType.TOTR)

    # 按视场 RSCE / 畸变
    # RSCE: P1=Sampling, P2=Wave, P3=Hx, P4=Hy
    # 畸变用 REAY/理想像高手动计算，比 DIST 操作数可靠：
    #   real_chief_y = REAY(img_surf, 0, 0, hy, 0, 0)  主光线真实像高
    #   ideal_y      = EFFL × tan(field_angle_rad)       理想像高（无畸变参考）
    #   distortion%  = (real_chief_y - ideal_y) / |ideal_y| × 100
    # DIST 操作数在 ZOS-API GetOperandValue 中所有视场返回同一全局值，不可按视场区分。
    rms_list, dist_list = [], []
    try:
        fields   = sys_.SystemData.Fields
        nf       = fields.NumberOfFields
        lde_m    = sys_.LDE
        img_surf = int(lde_m.NumberOfSurfaces - 1)
        effl_val = out.get("effl") or 0.0

        # 视场类型: 0=Angle(度), 1=ObjectHeight, 2=ParaxialImageHeight, 3=RealImageHeight
        try:
            field_type = int(fields.GetFieldType())
        except Exception:
            field_type = 0

        y_max = 0.0
        for fi in range(1, nf + 1):
            try:
                y_max = max(y_max, abs(float(fields.GetField(fi).Y)))
            except Exception:
                pass
        if y_max <= 0:
            y_max = 1.0

        for fi in range(1, nf + 1):
            try:
                y_i = float(fields.GetField(fi).Y)
            except Exception:
                y_i = 0.0
            hy = y_i / y_max

            rms = _gov(OpType.RSCE, 0, 0, 0.0, hy)
            rms_list.append(rms if rms is not None else 0.0)

            # 畸变：主光线真实像高 vs 理想像高
            try:
                real_chief_y = float(mfe.GetOperandValue(
                    OpType.REAY, img_surf, 0, 0.0, hy, 0.0, 0.0, 0.0, 0.0))

                if field_type == 0:
                    ideal_y = effl_val * math.tan(math.radians(abs(y_i)))
                else:
                    ideal_y = abs(y_i)

                if ideal_y < 1e-9:
                    dist_pct = 0.0
                else:
                    dist_pct = (real_chief_y - ideal_y) / ideal_y * 100.0

                if abs(dist_pct) > 200:
                    print(f"[DIAG metrics] field {fi} 畸变哨兵值({dist_pct:.1f}%)，置0")
                    dist_pct = 0.0

            except Exception as _de:
                print(f"[DIAG metrics] field {fi} 畸变计算失败: {_de}")
                dist_pct = 0.0

            dist_list.append(round(dist_pct, 6))

    except Exception as e:
        print(f"[DIAG metrics] 视场循环失败: {e}")

    out["rms_per_field_mm"]          = rms_list
    out["distortion_per_field_pct"]  = dist_list
    return out



# ═════════════════════════════════════════════════════════════
#   /status
# ═════════════════════════════════════════════════════════════
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"ok": True, "zemax_ready": True})


# ═════════════════════════════════════════════════════════════
#   /load_lens - 推送面型
# ═════════════════════════════════════════════════════════════
@app.route("/load_lens", methods=["POST"])
def load_lens():
    try:
        data  = request.json
        surfs = data.get("surfaces", [])
        fov   = float(data.get("fov",  30.0))
        fnum  = float(data.get("fnum",  2.8))
        stop_surface = data.get("stop_surface")   # 可选
        sys_  = _get_system()

        sys_.New(False)

        # ★★★ P8 致命修复 ★★★
        # sys_.New(False) 会重置 SystemData,包括 MaterialCatalogs。
        # 如果不重新 AddCatalog("CDGM"),后面 surf.Material = "H-ZF4A" 写入
        # 虽然不报错,但 Zemax 在追迹时查不到该玻璃 → 静默当成 n=1 空气
        # → EFFL = 1e10, WFNO = 1e4, 优化永远失败。
        # 这是 bridge 之前最大的坑,在此显式修掉。
        cdgm_ok, cdgm_note = _ensure_cdgm_catalog(sys_, label="load_lens")
        if not cdgm_ok:
            # 如果 CDGM 挂不上,后面写入 H-xxx 牌号会被当空气,直接返回错误让 agent 知道
            return jsonify({
                "error": "CDGM catalog 挂载失败,无法识别国产玻璃",
                "detail": cdgm_note,
                "hint": "检查 Zemax 安装目录下 Glasscat/CDGM.AGF 是否存在"
            }), 500

        # ★ Aperture 的设置移到最后 (写完 surfaces + 设完 stop 之后) ─────────
        # 原因: 在 stop 还没标记之前设 ImageSpaceFNum, Zemax 会把 LDE[1] 当光阑,
        # 近轴求解用错孔径 → EFFL=0. 先设成 Float by Stop 占位, 写完 surface
        # + 设好 STOP 后再一次性正式设置 Aperture.

        sys_.SystemData.Aperture.ApertureType = \
            ZOSAPI.SystemData.ZemaxApertureType.FloatByStopSize

        # ★★ 决定性修复：强制关闭 Afocal Image Space ★★
        # 如果这个 flag 是 True，Zemax 把整个系统当无焦系统，EFFL/WFNO 永远返回哨兵
        # (1e10 / 1e4)，MFE 所有空间量算不出，优化必然失败。默认值不可靠，必须显式置 False。
        try:
            afocal_before = sys_.SystemData.Aperture.AFocalImageSpace
            sys_.SystemData.Aperture.AFocalImageSpace = False
            afocal_after = sys_.SystemData.Aperture.AFocalImageSpace
            print(f"[DIAG load_lens] AFocalImageSpace: {afocal_before} -> {afocal_after}")
        except Exception as e:
            print(f"[DIAG load_lens] ⚠ AFocalImageSpace 设置失败: {e}")

        # ★★★ 决定性修复 #2：开启 Paraxial Ray Aiming ★★★
        # F/1.2 / F/1.4 这种大孔径 + STOP 嵌在透镜组中间的系统, Ray Aiming OFF 时
        # 近轴追迹无法瞄准真实 STOP, 结果 EFFL/WFNO 永远返回哨兵 1e10/1e4。
        # GUI 打开 .zmx 会自动启用 ray aiming, 所以 GUI 能算 EFFL=50; 但 ZOS-API
        # 的 LoadFile / 代码构建的 system 默认 OFF, 这是导致 bridge 和 GUI 结果
        # 不一致的直接原因。
        # 官方建议: "Paraxial Ray Aiming should always be used, should be turned on by default"
        try:
            RA = ZOSAPI.SystemData.RayAimingMethod
            ra_before = sys_.SystemData.RayAiming.RayAiming
            sys_.SystemData.RayAiming.RayAiming = RA.Paraxial
            ra_after = sys_.SystemData.RayAiming.RayAiming
            print(f"[DIAG load_lens] RayAiming: {ra_before} -> {ra_after}")
        except Exception as e:
            print(f"[DIAG load_lens] ⚠ RayAiming 设置失败: {e}")
            # 有的版本枚举路径不同，尝试 fallback
            try:
                sys_.SystemData.RayAiming.RayAiming = 1  # 1 = Paraxial in some versions
                print(f"[DIAG load_lens] RayAiming 通过整数 1 (Paraxial) 设置成功")
            except Exception as e2:
                print(f"[DIAG load_lens] ⚠ RayAiming 整数值也失败: {e2}")

        # Fields — 重构：用 SetNumberOfFields 确保3个视场真正建立
        fields = sys_.SystemData.Fields
        # ★ FieldType 必须显式设成 Angle，否则 Y 值被当 object height
        try:
            fields.SetFieldType(ZOSAPI.SystemData.FieldType.Angle)
            print(f"[DIAG load_lens] FieldType 设置为 Angle")
        except Exception as e:
            print(f"[DIAG load_lens] SetFieldType 失败: {e}")

        half = fov / 2.0
        # ★ FIX: DeleteAllFields 后 NumberOfFields=1，AddField 有时静默失败。
        # 正确做法：先 DeleteAllFields，再用 SetNumberOfFields(3) 预分配，
        # 最后逐一设每个视场的 Y 值。
        fields.DeleteAllFields()
        # 尝试 SetNumberOfFields 预分配3个视场
        _nf_ok = False
        try:
            fields.SetNumberOfFields(3)
            _nf_ok = True
            print(f"[DIAG load_lens] SetNumberOfFields(3) 成功")
        except Exception as e:
            print(f"[DIAG load_lens] SetNumberOfFields 失败({e})，改用 AddField 逐个添加")

        if _nf_ok:
            # 直接设置3个视场的 Y 值
            _field_y = [0.0, half * 0.7, half]  # ★ 设计指南: 0, 0.7×HFoV, HFoV
            for fi, yv in enumerate(_field_y, start=1):
                try:
                    f = fields.GetField(fi)
                    f.Y = yv; f.X = 0.0; f.Weight = 1.0
                    print(f"[DIAG load_lens] Field[{fi}] Y={yv:.3f} 设置成功")
                except Exception as e:
                    print(f"[DIAG load_lens] Field[{fi}] Y={yv:.3f} 设置失败: {e}")
        else:
            # 回退：Field#1 改 Y，再 AddField 两次，并验证
            f1 = fields.GetField(1)
            f1.Y = 0.0; f1.X = 0.0; f1.Weight = 1.0
            for yv in [half * 0.7, half]:  # ★ 设计指南: 0, 0.7×HFoV, HFoV
                try:
                    added = fields.AddField(0, yv, 1.0)
                    print(f"[DIAG load_lens] AddField(Y={yv:.3f}) -> {added}")
                except Exception as e:
                    print(f"[DIAG load_lens] AddField(Y={yv:.3f}) 失败: {e}")

        # 验证实际视场数量
        try:
            nf_actual = fields.NumberOfFields
            print(f"[DIAG load_lens] 视场数量验证: NumberOfFields={nf_actual}")
            for fi in range(1, nf_actual + 1):
                f = fields.GetField(fi)
                print(f"  Field[{fi}] Y={float(f.Y):.3f}")
            if nf_actual < 3:
                print(f"[DIAG load_lens] ⚠ 视场数={nf_actual}<3，场曲将无法被优化！")
        except Exception as e:
            print(f"[DIAG load_lens] 视场验证失败: {e}")

        # ── P4: 三波长 F-d-C 才看得出色差 ───────────────────────
        wl = sys_.SystemData.Wavelengths
        wl.SelectWavelengthPreset(
            ZOSAPI.SystemData.WavelengthPreset.FdC_Visible)

        # Surfaces
        # P修复: agent 传过来的 surfs 含 object(0) + 实物面(...) + image(末尾空壳)
        # Zemax New() 默认有 OBJ/STO/IMA 三面, 我们只写 object + 实物面,
        # image 让 Zemax 自己的最后那面充当, 否则多出的一面会打断到像的链路
        real_surfs = surfs[:-1] if surfs else surfs  # 去掉末尾 image 占位
        lde = sys_.LDE
        # 目标: LDE 最终有 len(real_surfs) + 1 面(=写入面 + image)
        while lde.NumberOfSurfaces < len(real_surfs) + 1:
            lde.InsertNewSurfaceAt(lde.NumberOfSurfaces - 1)

        for i, s in enumerate(real_surfs):
            surf = lde.GetSurfaceAt(i)
            r = s.get("radius", 0)
            t = s.get("thickness", 0)
            mat = s.get("material", "")
            # P修复: agent 传过来的 key 是 "sd"（有时是 "semi_diameter"）
            sd = s.get("semi_diameter", s.get("sd", None))
            _wrote_r = _wrote_t = _wrote_m = False
            _wrote_sd = False
            _mat_err = None

            # ★ Object 面特判: agent 用 ~10^6/1e10 表示"无穷远",我们必须用 float('inf')
            #   写入 surface.Thickness —— ZOS-API 会识别并挂上 Infinity solve
            #   （ZOSAPI.Editors.SolveType.Infinity 这个枚举在某些版本不存在,
            #    所以绕过 SolveType 那条路,直接赋值 float('inf') 最稳）。
            #   不用 1e10 浮点 fallback —— 那在 Zemax 里被当成"非常远但有限距离",
            #   近轴追迹 EFFL/WFNO 全部返回哨兵 1e10 / 1e4,导致 MFE 炸掉。
            if i == 0 and t is not None and abs(float(t)) > 1e5:
                obj_inf_ok = False
                # 首选: surface.Thickness = float('inf') —— ZOSPy 官方示例用法
                try:
                    surf.Thickness = float("inf")
                    obj_inf_ok = True
                    _wrote_t = True
                    t = float("inf")
                    print(f"[DIAG load_lens] ✓ Object thickness = inf "
                          f"(agent 原值={s.get('thickness')})")
                except Exception as e:
                    print(f"[DIAG load_lens] surf.Thickness=inf 失败: {e}")

                # 回退 1: ThicknessCell 挂 Infinity solve（老 API 枚举路径）
                if not obj_inf_ok:
                    try:
                        tcell = surf.ThicknessCell
                        solve_inf = tcell.CreateSolveType(
                            ZOSAPI.Editors.SolveType.Infinity)
                        tcell.SetSolveData(solve_inf)
                        obj_inf_ok = True
                        _wrote_t = True
                        t = float("inf")
                        print(f"[DIAG load_lens] ✓ Object thickness 用 Infinity solve")
                    except Exception as e:
                        print(f"[DIAG load_lens] Infinity solve 失败: {e}")

                # 回退 2: 都失败时才用大浮点 —— 但这会让 MFE 算不出 EFFL,优化必炸
                if not obj_inf_ok:
                    try:
                        surf.Thickness = 1.0e10
                        _wrote_t = True
                        t = 1.0e10
                        print(f"[DIAG load_lens] ⚠ (最后回退) Object thickness = 1e10 浮点 "
                              f"—— EFFL/WFNO 将返回哨兵值,优化会失败")
                    except Exception as e:
                        print(f"[DIAG load_lens] Object thickness 全部设法失败: {e}")

            if r and abs(r) < 1e8:
                surf.Radius = float(r)
                _wrote_r = True
            if t is not None and abs(t) < 1e10 and not _wrote_t:
                t_val = float(t)
                # ★ 修复: 负厚度在物理上不合理，写入时直接修正
                if t_val < 0:
                    mat_check = str(mat).upper() if mat else ""
                    t_fixed = 1.0 if mat_check not in ("", "AIR") else 0.5
                    print(f"[DIAG load_lens] surf[{i}] thickness={t_val:.3f} 负值，修正为 {t_fixed}")
                    t_val = t_fixed
                surf.Thickness = t_val
                _wrote_t = True
            # P修复: AIR / 空 需要显式清空 Material，而不是"跳过"
            mat_str = str(mat).upper() if mat else ""
            if mat_str in ("", "AIR"):
                try:
                    surf.Material = ""
                except Exception as e:
                    _mat_err = f"clear: {e}"
            else:
                try:
                    surf.Material = str(mat)
                    _wrote_m = True
                except Exception as e:
                    _mat_err = str(e)
            # ★ 设计指南: 净口径设置为自动计算（不要固定）
            # 先用数据库 sd 值写入固定值（确保光线能追到），再设 Automatic solve。
            # 若 Automatic solve 成功，Zemax 会在追迹后自动覆盖这个初始值。
            # 若失败（如 surf1 在 ImageSpaceFNum 模式下哨兵值=0.001），
            # 固定初始值保底，防止光线被截断。
            if sd is not None and float(sd) > 0:
                try:
                    surf.SemiDiameter = float(sd)
                    _wrote_sd = True
                except Exception:
                    pass
            try:
                sdcell = surf.SemiDiameterCell
                sol = sdcell.CreateSolveType(ZOSAPI.Editors.SolveType.Automatic)
                sdcell.SetSolveData(sol)
                _wrote_sd = True
                print(f"[DIAG load_lens] surf[{i}] SemiDiameter = Auto (初始值={sd})")
            except Exception as e:
                # Automatic 失败时保留上面写入的固定值
                if sd is not None and float(sd) > 0:
                    try:
                        surf.SemiDiameter = float(sd)
                        _wrote_sd = True
                        print(f"[DIAG load_lens] surf[{i}] SemiDiameter = {float(sd):.3f} (Fixed, Auto失败: {e})")
                    except Exception as e2:
                        _mat_err = (_mat_err or "") + f" SD_ERR={e2}"
            print(f"[DIAG load_lens] payload[{i}] r={r} t={t} mat='{mat}' sd={sd} "
                  f"→ wrote R={_wrote_r} T={_wrote_t} M={_wrote_m} SD={_wrote_sd}"
                  + (f" ERR={_mat_err}" if _mat_err else ""))

        # ── P7: Stop surface ─────────────────────────────────────
        # 用 payload 里的原始 sd 值来判断光阑（Automatic solve 会改变LDE的SD，不可靠）
        # 光阑面 = r≈∞ + 材料=AIR + payload sd 最小的面
        if stop_surface is None:
            _stop_cands = []
            for i, s in enumerate(surfs):
                if i == 0 or i == len(surfs) - 1:
                    continue
                r_ = s.get("radius", 0)
                m_ = str(s.get("material", "") or "").upper()
                sd_ = s.get("semi_diameter", s.get("sd", 0)) or 0
                is_inf = (r_ is None) or (abs(float(r_)) >= 1e8) or (abs(float(r_)) < 1e-3)
                if is_inf and m_ in ("", "AIR") and float(sd_) > 0:
                    _stop_cands.append((i, float(sd_)))
            if _stop_cands:
                stop_surface = min(_stop_cands, key=lambda x: x[1])[0]
                print(f"[DIAG load_lens] 自动识别 stop_surface = {stop_surface} (payload sd={min(_stop_cands, key=lambda x: x[1])[1]:.3f})")
            else:
                print("[DIAG load_lens] ⚠ 未识别到 stop_surface，Zemax 会用默认第一面")

        # ★ 关键: 在设置光阑之前，先把光阑面的 SD 强制设为 payload 里的原始值
        # Automatic solve 把光阑面 SD 算成了 16.88mm（错误），需要强制修正
        if stop_surface is not None:
            try:
                si = int(stop_surface)
                orig_sd = None
                if 0 <= si < len(surfs):
                    orig_sd = surfs[si].get("semi_diameter", surfs[si].get("sd", None))
                if orig_sd is not None and float(orig_sd) > 0:
                    lde.GetSurfaceAt(si).SemiDiameter = float(orig_sd)
                    print(f"[DIAG load_lens] 光阑面[{si}] SD 强制修正为 payload 值 {float(orig_sd):.3f}")
            except Exception as _e:
                print(f"[DIAG load_lens] 光阑面 SD 修正失败: {_e}")

        if stop_surface is not None:
            stop_set_ok = False
            stop_idx = int(stop_surface)

            # 方法 1（最可靠）: LDE.StopSurface（int 属性，系统级）
            try:
                lde.StopSurface = stop_idx
                # 立刻回读验证
                try:
                    actual = int(lde.StopSurface)
                    if actual == stop_idx:
                        stop_set_ok = True
                        print(f"[DIAG load_lens] ✓ LDE.StopSurface = {stop_idx} 设置成功")
                    else:
                        print(f"[DIAG load_lens] ⚠ LDE.StopSurface 写入后回读={actual}，不等于 {stop_idx}")
                except Exception:
                    # 写成功但无法回读，也先当成功
                    stop_set_ok = True
                    print(f"[DIAG load_lens] LDE.StopSurface = {stop_idx} 已写入（无法回读验证）")
            except Exception as e:
                print(f"[DIAG load_lens] LDE.StopSurface 赋值失败: {e}")

            # 方法 2（回退）: Surface.IsStop = True
            if not stop_set_ok:
                try:
                    surf_s = lde.GetSurfaceAt(stop_idx)
                    surf_s.IsStop = True
                    # 回读验证（某些版本 IsStop 是 get-only，赋值不报错但不生效）
                    try:
                        if bool(lde.GetSurfaceAt(stop_idx).IsStop):
                            stop_set_ok = True
                            print(f"[DIAG load_lens] ✓ (回退) Surface[{stop_idx}].IsStop=True 成功")
                        else:
                            print(f"[DIAG load_lens] ⚠ IsStop 赋值无效（get-only 或被忽略）")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[DIAG load_lens] (回退) IsStop 赋值异常: {e}")

            if not stop_set_ok:
                print(f"[DIAG load_lens] ❌ 光阑设置全部失败，Zemax 会把第一面当光阑 → "
                      f"系统孔径错误，后续 rms/effl/fnum 将为 0 或垃圾值")

        # ★ 现在 stop 和 surfaces 都就位了, 正式设置 Aperture ─────────────
        # ★ 设计指南: 孔径类型 = 像方空间F/# (ImageSpaceFNum)
        # 直接用 fnum 值, Zemax 自动反推入瞳大小, 无需手动计算 EPD.
        try:
            sys_.SystemData.Aperture.ApertureType = \
                ZOSAPI.SystemData.ZemaxApertureType.ImageSpaceFNum
            sys_.SystemData.Aperture.ApertureValue = float(fnum)
            print(f"[DIAG load_lens] ✓ Aperture = ImageSpaceFNum, F/#={fnum:.3f}")
        except Exception as e:
            print(f"[DIAG load_lens] ⚠ ImageSpaceFNum 设置失败, 退回 FloatByStopSize: {e}")
            try:
                sys_.SystemData.Aperture.ApertureType = \
                    ZOSAPI.SystemData.ZemaxApertureType.FloatByStopSize
            except Exception:
                pass


        # ── 诊断：load 完后把 LDE 里实际的内容 dump 一遍 ──
        print(f"[DIAG load_lens] 推送了 {len(surfs)} 面, "
              f"LDE 现在有 {lde.NumberOfSurfaces} 面 (含 image)")
        for i in range(lde.NumberOfSurfaces):
            try:
                s_ = lde.GetSurfaceAt(i)
                r_ = s_.Radius
                t_ = s_.Thickness
                m_ = s_.Material
                sd_ = s_.SemiDiameter
                try:
                    stop_flag = " STOP" if s_.IsStop else ""
                except Exception:
                    stop_flag = ""
                print(f"  LDE[{i}] R={r_:.4g} T={t_:.4g} "
                      f"mat='{m_}' SD={sd_:.4g}{stop_flag}")
            except Exception as e:
                print(f"  LDE[{i}] 读取失败: {e}")
        # Aperture / Fields / Wavelength 实际值
        try:
            ap = sys_.SystemData.Aperture
            print(f"[DIAG load_lens] Aperture type={ap.ApertureType} "
                  f"value={ap.ApertureValue}")
        except Exception as e:
            print(f"[DIAG load_lens] 读 Aperture 失败: {e}")
        try:
            nf = sys_.SystemData.Fields.NumberOfFields
            print(f"[DIAG load_lens] Fields: {nf} 个")
            for fi in range(1, nf + 1):
                f = sys_.SystemData.Fields.GetField(fi)
                print(f"  Field[{fi}] X={f.X} Y={f.Y} W={f.Weight}")
        except Exception as e:
            print(f"[DIAG load_lens] 读 Fields 失败: {e}")
        try:
            nw = sys_.SystemData.Wavelengths.NumberOfWavelengths
            print(f"[DIAG load_lens] Wavelengths: {nw} 个")
        except Exception as e:
            print(f"[DIAG load_lens] 读 Wavelengths 失败: {e}")

        # ★ 保存 .zmx 到 C:\zemax_debug\current.zmx, 这样可以用 GUI 打开看
        #   Zemax 会弹红色错误,告诉我们哪个面/参数不对
        try:
            os.makedirs(r"C:\zemax_debug", exist_ok=True)
            save_path = r"C:\zemax_debug\current.zmx"
            sys_.SaveAs(save_path)
            print(f"[DIAG load_lens] ✓ 系统已保存到 {save_path} "
                  f"(可用 Zemax GUI 打开检查)")
        except Exception as e:
            print(f"[DIAG load_lens] ⚠ 保存 .zmx 失败: {e}")

        # ★ 直接用 MFE 的 REAY operand 查"边缘光在像面上的 Y 坐标"。
        #   原先用 BatchRayTrace + ReadNextResult 解析字段,但 ZOS-API 各版本
        #   返回元组结构不一致(X/Y/Z 的索引在不同版本里漂移), 用 REAY 从
        #   MFE.GetOperandValue 直接取值最稳 —— 和 Zemax GUI 里打 REAY 算子
        #   完全等价。
        #
        #   REAY(Surf, Wave, Hx, Hy, Px, Py):
        #     Surf: 目标面号(0=image), Wave: 波长编号(0=默认)
        #     Hx/Hy: 归一化视场坐标; Px/Py: 归一化光瞳坐标
        #
        #   探针逻辑: 轴上 Py=0 时 Y 应该 ≈0, 而 Py=0.8 的边缘光若折射正常,
        #   也会接近聚焦点(小值); 若 Y ≈ 光瞳半径则说明光线几乎直透=玻璃失效.
        try:
            OpType = ZOSAPI.Editors.MFE.MeritOperandType
            mfe_probe = sys_.MFE
            # image surface = NumberOfSurfaces - 1 (0-indexed)
            img_surf = int(lde.NumberOfSurfaces - 1)

            # 轴上光 Py=0 (应该 Y≈0)
            y_axis = mfe_probe.GetOperandValue(
                OpType.REAY, img_surf, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            # 边缘光 Py=0.8 (光学正常时 Y 也应该小,因为光线要聚焦)
            y_marg = mfe_probe.GetOperandValue(
                OpType.REAY, img_surf, 0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0)

            print(f"[DIAG load_lens] REAY 探针 (像面 Y): "
                  f"chief(Py=0)={float(y_axis):.4g}, marginal(Py=0.8)={float(y_marg):.4g}")

            # 健康检查:正常系统 |y_marg| 应该 << STOP_SD.
            # 若 |y_marg| 与 STOP_SD × 0.8 同量级 → 光线直透,玻璃没命中
            try:
                stop_sd_check = float(lde.GetSurfaceAt(
                    int(stop_surface) if stop_surface else 1
                ).SemiDiameter or 10.0)
            except Exception:
                stop_sd_check = 10.0

            abs_marg = abs(float(y_marg))
            if abs_marg > 0.5 * stop_sd_check:
                print(f"[DIAG load_lens] ❌❌ 边缘光 |Y|={abs_marg:.3g} "
                      f"≈ STOP_SD×0.8={stop_sd_check*0.8:.3g}, "
                      f"几乎没有折射! 很可能玻璃 catalog 未命中.")
            else:
                print(f"[DIAG load_lens] ✓ 边缘光 |Y|={abs_marg:.3g} "
                      f"远小于 STOP_SD={stop_sd_check:.3g}, 折射正常")
        except Exception as e:
            print(f"[DIAG load_lens] REAY 探针失败: {e}")

        # ═══════════════════════════════════════════════════════════════
        # ★★★ 三路 EFFL 探针 - 定位 MFE 为什么返回 1e10 哨兵 ★★★
        #
        # 路径 A: LDE.GetFirstOrderData() -- 直接从一阶数据拿,完全绕开 MFE
        # 路径 B: MFE.GetOperandValue(EFFL, 0,0,0,0,0,0,0,0) -- 我们正在用的
        # 路径 C: SaveAs → LoadFile → 重新读 EFFL -- 如果这条对了, 内存脏必须洗
        #
        # 3 条至少有一条正常时, 我们就知道从哪条拿 EFFL, 并判断出脏点在哪层
        # ═══════════════════════════════════════════════════════════════
        print("[DIAG probe] ═══ 三路 EFFL 探针 ═══")

        # --- 路径 A: LDE.GetFirstOrderData ---
        try:
            fod = sys_.LDE.GetFirstOrderData()
            # 返回 tuple: (EFL, real_wF#, paraxial_wF#, paraxial_img_h, paraxial_mag)
            print(f"[DIAG probe] A. LDE.GetFirstOrderData() -> {fod}")
            if fod is not None and len(fod) >= 3:
                print(f"[DIAG probe]    EFL={fod[0]:.4f}, "
                      f"real_W-F#={fod[1]:.4f}, paraxial_W-F#={fod[2]:.4f}")
        except Exception as e:
            print(f"[DIAG probe] A. GetFirstOrderData 失败: {e}")
            # 有些 ZOS-API 版本签名不同，逐个尝试
            try:
                from System import Double
                out = [Double(0)] * 5
                ok = sys_.LDE.GetFirstOrderData(out[0], out[1], out[2], out[3], out[4])
                print(f"[DIAG probe] A'. GetFirstOrderData(out[5]) -> ok={ok}, vals={[float(x) for x in out]}")
            except Exception as e2:
                print(f"[DIAG probe] A'. GetFirstOrderData(out) 也失败: {e2}")

        # --- 路径 B: MFE.GetOperandValue 再试一次，记录全参数 ---
        try:
            OpType = ZOSAPI.Editors.MFE.MeritOperandType
            mfe = sys_.MFE
            v_effl = mfe.GetOperandValue(OpType.EFFL, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_wfno = mfe.GetOperandValue(OpType.WFNO, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            v_totr = mfe.GetOperandValue(OpType.TOTR, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            print(f"[DIAG probe] B. MFE.GetOperandValue -> "
                  f"EFFL={v_effl}, WFNO={v_wfno}, TOTR={v_totr}")
        except Exception as e:
            print(f"[DIAG probe] B. MFE.GetOperandValue 失败: {e}")

        # --- 路径 C: SaveAs + LoadFile 洗一遍内存 ---
        try:
            refresh_path = r"C:\zemax_debug\_refresh_probe.zmx"
            sys_.SaveAs(refresh_path)
            # LoadFile 第 2 个参数是 saveIfNeeded
            load_ok = sys_.LoadFile(refresh_path, False)
            print(f"[DIAG probe] C. SaveAs+LoadFile ok={load_ok}")
            try:
                OpType = ZOSAPI.Editors.MFE.MeritOperandType
                v_effl_c = sys_.MFE.GetOperandValue(OpType.EFFL, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                v_wfno_c = sys_.MFE.GetOperandValue(OpType.WFNO, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                print(f"[DIAG probe] C. reload 后 MFE -> EFFL={v_effl_c}, WFNO={v_wfno_c}")
            except Exception as e:
                print(f"[DIAG probe] C. reload 后读 MFE 失败: {e}")
            # 再读一次 GetFirstOrderData 对比
            try:
                fod_c = sys_.LDE.GetFirstOrderData()
                print(f"[DIAG probe] C. reload 后 GetFirstOrderData -> {fod_c}")
            except Exception as e:
                print(f"[DIAG probe] C. reload 后 GetFirstOrderData 失败: {e}")
        except Exception as e:
            print(f"[DIAG probe] C. SaveAs+LoadFile 失败: {e}")

        print("[DIAG probe] ═══ 探针结束 ═══")

        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════
#   /metrics - 真值指标（供 agent 的 check_spec 用）   [P3 新增]
# ═════════════════════════════════════════════════════════════
@app.route("/metrics", methods=["GET", "POST"])
def metrics():
    try:
        sys_ = _get_system()
        m = _read_system_metrics(sys_)
        return jsonify(m)
    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════
#   /zemax_optimize  —— 真正跑 DLS（修 P0+P1+P2）
# ═════════════════════════════════════════════════════════════
@app.route("/zemax_optimize", methods=["POST"])
def zemax_optimize():
    try:
        payload = request.json or {}
        # ★ P17: cycles<=0 或缺省 => Automatic(跑到收敛)
        cycles  = int(payload.get("cycles", -1))
        target_effl = payload.get("target_effl", None)
        target_fnum = payload.get("target_fnum", None)

        sys_ = _get_system()
        lde  = sys_.LDE
        n    = lde.NumberOfSurfaces

        # ── P0: 把所有透镜面的 radius 和 thickness 标为变量 ────────────
        # ★ 设计指南: 除了光阑面之外，所有透镜面的厚度和曲率半径设置为变量
        # radius: 仅有限曲率的玻璃面（平面/空气面跳过）
        # thickness: 所有非光阑面、非物面、非像面的面（玻璃+空气间隔均包含）
        #            有 MNCA/MNCG 约束保护，不会穿插
        SolveType_ns = ZOSAPI.Editors.SolveType
        var_count = 0
        t_var_count = 0
        # 读取光阑面号，避免把光阑面厚度也标变量
        try:
            stop_surf_idx = int(lde.StopSurface)
        except Exception:
            stop_surf_idx = None

        # ★ 修复1: 仅修复哨兵值（SD < 0.5mm），不强制改动正常值
        # Automatic solve 失败时 Zemax 给 0.001 等哨兵值，这类才需要修复。
        # 注意: 不要根据 EPD 强制修改 SD，那样会破坏 Automatic solve 的逻辑。
        try:
            for i in range(1, n - 1):
                surf_chk = lde.GetSurfaceAt(i)
                sd_chk = float(surf_chk.SemiDiameter or 0)
                if sd_chk < 0.5:
                    surf_chk.SemiDiameter = 10.0
                    print(f"[FIX-SD] surf[{i}] SemiDiameter {sd_chk:.4f} → 10.0 (哨兵值修复)")
        except Exception as e:
            print(f"[FIX-SD] SD修复失败（忽略）: {e}")

        # ★ 预处理: 把玻璃厚度不足 6mm 的面调整到 6mm，满足优化向导约束
        # 原因: MNCG=6mm 是工程标准，但数据库镜头缩放后玻璃可能 <6mm，
        #       直接优化会让优化器把所有自由度都用来满足厚度约束而不是优化像质
        try:
            for i in range(1, n - 1):
                s_pre = lde.GetSurfaceAt(i)
                mat_pre = str(s_pre.Material or "").upper()
                if mat_pre in ("", "AIR"):
                    continue
                t_pre = float(s_pre.Thickness or 0)
                if 0 < t_pre < 2.0:
                    s_pre.Thickness = 2.0
                    print(f"[PRE-T] surf[{i}] 玻璃厚度 {t_pre:.3f} → 2.0 (满足MNCG=2mm约束，与target一致)")
        except Exception as e:
            print(f"[PRE-T] 玻璃厚度预处理失败: {e}")

        # ★ 修复2: 修复负厚度（优化跑偏产生的物理非法面）
        # MNCA/MNCG 约束在优化过程中是软约束，有时优化器仍会产生负厚度。
        # 在下一轮优化前强制修正，防止追迹完全失败。
        try:
            for i in range(1, n - 1):
                surf_chk = lde.GetSurfaceAt(i)
                t_chk = float(surf_chk.Thickness or 0)
                mat_chk = str(surf_chk.Material or "").upper()
                if t_chk < 0:
                    fixed_t = 1.0 if mat_chk not in ("", "AIR") else 0.5
                    surf_chk.Thickness = fixed_t
                    print(f"[FIX-T] surf[{i}] Thickness {t_chk:.4f} → {fixed_t} (负厚度已修复)")
        except Exception as e:
            print(f"[FIX-T] 负厚度修复失败（忽略）: {e}")

        for i in range(n):
            surf = lde.GetSurfaceAt(i)
            mat  = str(surf.Material or "").upper()
            r    = surf.Radius
            is_stop = (stop_surf_idx is not None and i == stop_surf_idx)
            # ① radius 变量：非物面(0)、非像面(n-1)、非光阑面、有限曲率的所有面
            # ★ 设计指南: 除光阑面外所有透镜面的曲率半径设为变量
            # 修复：之前只对玻璃面设，空气面（镜片后面）的radius也应是变量
            if not (i == 0 or i == n - 1 or is_stop):
                if r is not None and 1e-3 < abs(r) < 1e8:
                    try:
                        cell = surf.RadiusCell
                        solve = cell.CreateSolveType(SolveType_ns.Variable)
                        cell.SetSolveData(solve)
                        var_count += 1
                    except Exception as e:
                        print(f"set radius var surf{i} fail: {e}")
            # ② thickness 变量：★ 只有空气间隔设为变量，玻璃厚度固定不动
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 根本原因分析（沙漏形镜片）:
            #   玻璃厚度设为变量时，CTGT(weight=100)/MNCG(weight=50) 是软约束，
            #   DLS 降 RMS 的梯度力足以绕开两者，把中心厚度压到 ~0，
            #   镜片退化为"沙漏/蝴蝶结"形，上下三角超出镜筒，物理不可加工。
            # 解决方案:
            #   玻璃厚度只用于初始结构（由数据库或 PRE-T 给出），不参与优化。
            #   优化自由度 = 所有曲率半径 + 所有空气间隔（已足够丰富）。
            #   空气间隔变化对像差平衡贡献大，玻璃厚度变化对像差贡献相对小。
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # BFFL (n-2): MarginalRayHeight solve controls this, not Variable
            # Note: stop surface THICKNESS (air gap after stop) IS a variable
            #       only stop RADIUS is fixed (flat surface, already skipped above)
            if i == 0 or i == n - 1 or i == n - 2:
                continue
            t = surf.Thickness
            if t is None or abs(float(t)) > 1e5:
                continue
            # ★ 关键门控：玻璃面厚度不设为变量
            _mat_for_t = str(surf.Material or "").upper()
            if _mat_for_t not in ("", "AIR"):
                # 玻璃厚度固定，跳过，不设 Variable
                continue
            # 空气间隔设为变量（可自由改变，靠 MNCA+CTGT 保护下限）
            try:
                cell = surf.ThicknessCell
                solve = cell.CreateSolveType(SolveType_ns.Variable)
                cell.SetSolveData(solve)
                t_var_count += 1
                # 尝试设硬边界（部分版本支持）
                try:
                    solve_data = cell.GetSolveData()
                    solve_data.Minimum = 0.5   # 空气最小 0.5mm
                    solve_data.Maximum = 200.0
                    cell.SetSolveData(solve_data)
                except Exception:
                    pass
            except Exception as e:
                print(f"set air thickness var surf{i} fail: {e}")
        print(f"[DIAG optimize] 设置了 {var_count} 个 radius 变量, {t_var_count} 个 thickness 变量")
        if var_count == 0 and t_var_count == 0:
            return jsonify({"error": "没有可设为变量的面（全是平面或空气？）"}), 400

        # ★ 预处理: 优化前先跑 QuickFocus，把系统调到可追迹状态
        # 原因: 数据库镜头可能有负厚度，导致追迹失败，DLS在无效状态下乱跑。
        # QuickFocus 只调整像面位置（不改变镜片），让系统先能正常追迹再优化。
        try:
            qf_tool = sys_.Tools.OpenQuickFocus()
            if qf_tool is not None:
                qf_tool.Criterion = ZOSAPI.Tools.General.QuickFocusCriterion.SpotSizeRadial
                qf_tool.UseCentroid = True
                qf_tool.RunAndWaitForCompletion()
                qf_tool.Close()
                print(f"[DIAG optimize] ✓ QuickFocus 完成，系统已预聚焦")
        except Exception as e:
            print(f"[DIAG optimize] ⚠ QuickFocus 失败（忽略）: {e}")

        # ★ 关键: 给最后一个实物面的 thickness 设 MarginalRayHeight solve（目标高度=0）
        # 这样 Zemax 优化时会自动找到焦点位置（像面自动跟随），否则像面固定不动，光线无法汇聚。
        try:
            last_real = n - 2  # n-1 是像面，n-2 是最后一个实物面
            surf_last = lde.GetSurfaceAt(last_real)
            cell_last = surf_last.ThicknessCell
            mrh_solve = cell_last.CreateSolveType(SolveType_ns.MarginalRayHeight)
            try:
                mrh_solve.Height = 0.0
                mrh_solve.PupilZone = 1.0
            except Exception:
                pass
            cell_last.SetSolveData(mrh_solve)
            print(f"[DIAG optimize] ✓ surf[{last_real}] thickness → MarginalRayHeight solve (像面自动聚焦)")
        except Exception as e:
            try:
                cell_last = lde.GetSurfaceAt(n - 2).ThicknessCell
                cell_last.SetSolveData(cell_last.CreateSolveType(SolveType_ns.Variable))
                print(f"[DIAG optimize] ⚠ MarginalRayHeight 不可用，surf[{n-2}] thickness → Variable")
            except Exception as e2:
                print(f"[DIAG optimize] ⚠ 像面 solve 设置失败: {e} / {e2}")

        # ── P2: Merit Function ──────────────────────────────────
        # ★ 新流程：先用 Optimization Wizard 自动生成标准操作数，
        #   再手动追加 EFFL / TOTR（向导不生成这两个）。
        #
        # Wizard 负责生成:
        #   RSCE（各视场像质）、MNCA/MNCG/MXCG/MNET（厚度约束）、
        #   CTGT（中心厚度下限）、DIMX（畸变）、FCUR（场曲）、MXSD（净口径）
        # 手动追加:
        #   EFFL（焦距锚定）、TOTR（总长上限）、WFNO（F#）、BFLD（后焦距）
        #
        mf = sys_.MFE
        MFEcol = ZOSAPI.Editors.MFE.MeritColumn   # 供后面手动追加用

        # -- Step 1: SEQOptimizationWizard2 (confirmed API, 2024 R1)
        # Properties from diag:
        #   Type=RMS, Criterion=Wavefront->Spot(1), Reference=Centroid
        #   UseGaussianQuadrature, GridSizeNxN, Rings
        #   UseAirBoundaryValues,   AirMin/AirMax/AirEdgeThickness
        #   UseGlassBoundaryValues, GlassMin/GlassMax/GlassEdgeThickness
        #   UseMaximumDistortion,   MaxDistortionPct
        #   Apply() / OK()
        _wizard_ok = False
        try:
            wiz = mf.SEQOptimizationWizard2

            # Criterion: Spot (Python.NET 3.0 needs Enum class, not int)
            try:
                wiz.Criterion = wiz.Criterion.__class__(1)  # 1=Spot
                print("[DIAG wizard] Criterion ->", wiz.Criterion)
            except Exception as _e:
                print("[DIAG wizard] Criterion set failed:", _e)

            # Type=RMS(0), Reference=Centroid(0) keep defaults
            wiz.UseGaussianQuadrature = True
            wiz.GridSizeNxN = 3
            wiz.Rings = 3
            wiz.UseAllFields = True
            wiz.AssumeAxialSymmetry = True

            # Air boundary (checkbox + values)
            wiz.UseAirBoundaryValues = True
            wiz.AirMin = 0.5
            wiz.AirMax = 1000.0
            wiz.AirEdgeThickness = 0.0
            print("[DIAG wizard] Air: checked=", wiz.UseAirBoundaryValues, "Min=", wiz.AirMin)

            # Glass boundary (checkbox + values)
            wiz.UseGlassBoundaryValues = True
            wiz.GlassMin = 2.0
            wiz.GlassMax = 10.0
            wiz.GlassEdgeThickness = 1.0
            print("[DIAG wizard] Glass: checked=", wiz.UseGlassBoundaryValues,
                  "Min=", wiz.GlassMin, "Max=", wiz.GlassMax, "Edge=", wiz.GlassEdgeThickness)

            # Distortion 5%
            wiz.UseMaximumDistortion = True
            wiz.MaxDistortionPct = 5.0

            # SaveSettings: GUI opens wizard with these settings next time
            try:
                wiz.SaveSettings()
                print("[DIAG wizard] SaveSettings() OK")
            except Exception as _se:
                print("[DIAG wizard] SaveSettings failed:", _se)

            # Apply -> write operands to MFE
            wiz.Apply()
            _n_wiz = mf.NumberOfOperands
            _wizard_ok = True
            print("[DIAG optimize] Wizard Apply() done, operands:", _n_wiz)

        except Exception as _wiz_err:
            print("[DIAG optimize] SEQOptimizationWizard2 failed:", _wiz_err)
            _wizard_ok = False

        # ── Step 2: Wizard 失败时手动清空并重建（保底） ────────────
        if not _wizard_ok:
            _safety = 0
            while mf.NumberOfOperands > 0 and _safety < 500:
                try:
                    mf.RemoveOperandAt(1)
                except Exception:
                    break
                _safety += 1

            # 手动补 RSCE（各视场）
            num_fields = sys_.SystemData.Fields.NumberOfFields
            fields_obj = sys_.SystemData.Fields
            y_max = 0.0
            for fi in range(1, num_fields + 1):
                try:
                    y_max = max(y_max, abs(float(fields_obj.GetField(fi).Y)))
                except Exception:
                    pass
            if y_max <= 0:
                y_max = 1.0
            for fi in range(1, num_fields + 1):
                try:
                    y_i = float(fields_obj.GetField(fi).Y)
                except Exception:
                    y_i = 0.0
                hy = y_i / y_max
                op = mf.AddOperand()
                op.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.RSCE)
                op.GetOperandCell(MFEcol.Param1).IntegerValue = 0
                op.GetOperandCell(MFEcol.Param2).IntegerValue = 0
                op.GetOperandCell(MFEcol.Param3).DoubleValue  = 0.0
                op.GetOperandCell(MFEcol.Param4).DoubleValue  = hy
                op.Target = 0.0
                op.Weight = 4.0 if abs(hy) < 0.3 else (5.0 if abs(hy) < 0.85 else 6.0)
                print(f"[DIAG optimize] [fallback] RSCE field#{fi} Hy={hy:.3f} weight={op.Weight}")

            # 手动补厚度约束（向导功能的子集）
            for i in range(1, n - 1):
                try:
                    s_i  = lde.GetSurfaceAt(i)
                    mat_i = str(s_i.Material or "").upper()
                    if mat_i in ("", "AIR"):
                        op_a = mf.AddOperand()
                        op_a.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.MNCA)
                        op_a.GetOperandCell(MFEcol.Param1).IntegerValue = i
                        op_a.GetOperandCell(MFEcol.Param2).IntegerValue = 0
                        op_a.Target = 0.5
                        op_a.Weight = 50.0
                    else:
                        op_g = mf.AddOperand()
                        op_g.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.MNCG)
                        op_g.GetOperandCell(MFEcol.Param1).IntegerValue = i
                        op_g.GetOperandCell(MFEcol.Param2).IntegerValue = 0
                        op_g.Target = 2.0
                        op_g.Weight = 50.0
                        op_mx = mf.AddOperand()
                        op_mx.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.MXCG)
                        op_mx.GetOperandCell(MFEcol.Param1).IntegerValue = i
                        op_mx.GetOperandCell(MFEcol.Param2).IntegerValue = 0
                        op_mx.Target = 10.0
                        op_mx.Weight = 1.0
                        op_me = mf.AddOperand()
                        op_me.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.MNET)
                        op_me.GetOperandCell(MFEcol.Param1).IntegerValue = i
                        op_me.Target = 1.0
                        op_me.Weight = 20.0
                except Exception as _fb_e:
                    print(f"[DIAG optimize] [fallback] surf[{i}] 约束添加失败: {_fb_e}")
            print(f"[DIAG optimize] [fallback] 手动 MFE 构建完成，共 {mf.NumberOfOperands} 个操作数")

        # ─────────────────────────────────────────────────────────
        # ── Step 3: 手动追加 EFFL / BFLD / WFNO / TOTR ──────────
        # （向导不生成这四个，每次都无条件追加）
        # ─────────────────────────────────────────────────────────

        # （向导不生成这四个，每次都无条件追加）

        # 3a. EFFL 有效焦距约束（始终加入，用户值优先，否则当前值软锚）
        # weight=1：像质(RSCE)主导，EFFL 只作轻约束
        try:
            OpType_effl = ZOSAPI.Editors.MFE.MeritOperandType
            cur_effl = float(sys_.MFE.GetOperandValue(
                OpType_effl.EFFL, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) or 0)
            effl_target_val = float(target_effl) if (target_effl is not None and float(target_effl) > 0) \
                              else (cur_effl if 1.0 < abs(cur_effl) < 1e8 else 0)
            if effl_target_val > 0:
                op_effl = mf.AddOperand()
                op_effl.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.EFFL)
                op_effl.Target = effl_target_val
                op_effl.Weight = 1.0
                print(f"[DIAG optimize] EFFL 约束加入: target={effl_target_val:.3f}mm weight=1.0")
        except Exception as e:
            print(f"[DIAG optimize] ⚠ EFFL 操作数添加失败: {e}")

        # 3b. 后焦距约束 BFLD（防像面推到透镜内部，依次尝试 BFLD → REAR → CTGT）
        _bfld_added = False

        # ── 方案 A: BFLD ──────────────────────────────────────────────────
        try:
            op_bfld = mf.AddOperand()
            op_bfld.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.BFLD)
            try:
                cur_bfld = float(sys_.MFE.GetOperandValue(
                    ZOSAPI.Editors.MFE.MeritOperandType.BFLD,
                    0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) or 0)
                bfld_target = max(cur_bfld, 0.5) if 0 < cur_bfld < 1e8 else 5.0
            except Exception:
                bfld_target = 5.0
            op_bfld.Target = bfld_target
            op_bfld.Weight = 1.0
            _bfld_added = True
            print(f"[DIAG optimize] BFLD 后焦距约束加入: target={bfld_target:.3f}mm weight=1.0")
        except Exception:
            pass   # 此版本 ZOS-API 不支持 BFLD，尝试下一方案

        # ── 方案 B: REAR（部分版本的后焦距操作数）────────────────────────
        if not _bfld_added:
            try:
                op_rear = mf.AddOperand()
                op_rear.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.REAR)
                op_rear.Target = 5.0
                op_rear.Weight = 1.0
                _bfld_added = True
                print(f"[DIAG optimize] REAR 后焦距约束加入: target=5.0mm weight=1.0")
            except Exception:
                pass

        # ── 方案 C: CTGT 最后一个空气间隔下限 ≥ 0.5mm ───────────────────
        if not _bfld_added:
            try:
                last_air = n - 2   # 最后一个实物面（像面前一面）
                op_ct = mf.AddOperand()
                op_ct.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.CTGT)
                op_ct.GetOperandCell(ZOSAPI.Editors.MFE.MeritColumn.Param1).IntegerValue = last_air
                op_ct.Target = 0.5
                op_ct.Weight = 50.0
                _bfld_added = True
                print(f"[DIAG optimize] CTGT surf[{last_air}] 后焦距下限 ≥ 0.5mm (BFLD/REAR 均不可用)")
            except Exception as e:
                print(f"[DIAG optimize] ⚠ 后焦距约束全部失败，像面可能漂移: {e}")

        # 3c. WFNO F# 约束 ── 始终加入（无条件）
        # 若 payload 有 target_fnum 用指定值；否则读系统当前 F# 作软锚。
        # weight=2：保证光圈不在优化中漂移。
        try:
            _wfno_target = None
            if target_fnum is not None and float(target_fnum) > 0:
                _wfno_target = float(target_fnum)
            else:
                try:
                    _wfno_target = float(sys_.MFE.GetOperandValue(
                        ZOSAPI.Editors.MFE.MeritOperandType.WFNO,
                        0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) or 0)
                    if _wfno_target <= 0 or _wfno_target > 100:
                        _wfno_target = None
                except Exception:
                    _wfno_target = None
            if _wfno_target is not None:
                op_wfno = mf.AddOperand()
                op_wfno.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.WFNO)
                op_wfno.Target = _wfno_target
                op_wfno.Weight = 2.0
                print(f"[DIAG optimize] WFNO 约束加入: target=F/{_wfno_target:.2f} weight=2.0")
            else:
                print(f"[DIAG optimize] WFNO 无法读取当前值，跳过")
        except Exception as e:
            print(f"[DIAG optimize] ⚠ WFNO 添加失败: {e}")

        # 3d. TOTR 单边上限约束（OPLT）
        # OPLT = "Operand P Less Than"，只在 TOTR > max_totr 时产生惩罚（真正单边上限）。
        # 步骤：先加 TOTR 操作数（weight=0，仅用于测量），再加 OPLT 引用该行。
        # max_totr 来源优先级：① payload max_totr（RAG镜头TOTR）② 当前系统TOTR软锚
        try:
            target_totr = payload.get("target_totr", None)
            max_totr    = payload.get("max_totr", None)

            # 读当前系统 TOTR 作备用
            try:
                _cur_totr = float(sys_.MFE.GetOperandValue(
                    ZOSAPI.Editors.MFE.MeritOperandType.TOTR,
                    0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) or 0)
                if _cur_totr <= 0 or _cur_totr > 1e8:
                    _cur_totr = None
            except Exception:
                _cur_totr = None

            # 确定上限值
            if target_totr is not None and float(target_totr) > 0:
                _totr_max = float(target_totr)
                _totr_label = "用户精确目标"
            elif max_totr is not None and float(max_totr) > 0:
                _totr_max = float(max_totr)   # RAG镜头原始TOTR，不加弹性（严格上限）
                _totr_label = "RAG镜头TOTR上限"
            elif _cur_totr is not None:
                _totr_max = _cur_totr * 1.05  # 当前值+5%弹性作软锚
                _totr_label = "当前系统软锚"
            else:
                _totr_max = 200.0
                _totr_label = "默认宽松上限"

            # Step 1: 加 TOTR 测量操作数（weight=0，只用于 OPLT 引用）
            op_totr_meas = mf.AddOperand()
            op_totr_meas.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.TOTR)
            op_totr_meas.Weight = 0.0   # 不直接参与评价函数
            _totr_row = mf.NumberOfOperands  # 当前行号

            # Step 2: 加 OPLT 单边上限操作数（仅 TOTR > max 时惩罚）
            try:
                op_oplt = mf.AddOperand()
                op_oplt.ChangeType(ZOSAPI.Editors.MFE.MeritOperandType.OPLT)
                op_oplt.GetOperandCell(MFEcol.Param1).IntegerValue = _totr_row  # 指向 TOTR 行
                op_oplt.Target = _totr_max
                op_oplt.Weight = 2.0
                print(f"[DIAG optimize] TOTR OPLT 单边上限: row={_totr_row} max={_totr_max:.1f}mm ({_totr_label}) weight=2.0")
            except Exception as _oplt_e:
                # OPLT 不可用时退回对称 TOTR
                print(f"[DIAG optimize] OPLT 不可用({_oplt_e})，退回对称 TOTR 约束")
                op_totr_meas.Target = _totr_max
                op_totr_meas.Weight = 2.0
                print(f"[DIAG optimize] TOTR 对称约束: target={_totr_max:.1f}mm weight=2.0")
        except Exception as e:
            print(f"[DIAG optimize] TOTR/OPLT 添加失败: {e}")
        # ── P1/P17: Cycles 枚举映射 ─────────────────────────────
        # 实际枚举（OpticStudio 2024）: Automatic, Fixed_1_Cycle, Fixed_5_Cycles,
        # Fixed_10_Cycles, Fixed_50_Cycles, Infinite
        # P17: 约定 cycles=-1/0 表示 Automatic(跑到收敛),让 agent 不必猜轮数
        CycEnum = ZOSAPI.Tools.Optimization.OptimizationCycles
        fixed_map = {
            1:  CycEnum.Fixed_1_Cycle,
            5:  CycEnum.Fixed_5_Cycles,
            10: CycEnum.Fixed_10_Cycles,
            50: CycEnum.Fixed_50_Cycles,
        }
        if cycles is None or cycles <= 0:
            cycles_enum = CycEnum.Automatic
            closest     = "auto"
            print(f"[DIAG optimize] Cycles=Automatic (agent 传 cycles={cycles})")
        else:
            closest     = min(fixed_map.keys(), key=lambda k: abs(k - cycles))
            cycles_enum = fixed_map[closest]
            print(f"[DIAG optimize] Cycles={closest} (agent 传 cycles={cycles})")

        AlgEnum = ZOSAPI.Tools.Optimization.OptimizationAlgorithm
        opt = sys_.Tools.OpenLocalOptimization()
        if opt is None:
            return jsonify({"error": "OpenLocalOptimization 返回 None"}), 500

        opt.Algorithm = AlgEnum.DampedLeastSquares
        opt.Cycles    = cycles_enum

        # ── 诊断：优化前逐行 dump，value 用 GetOperandValue 独立求一次 ──
        # op.Value 在 interactive 模式下可能不随 CalculateMeritFunction 刷新
        # 所以同时打 cached (op.Value) 和 fresh (GetOperandValue)，方便排障
        try:
            mf.CalculateMeritFunction()
            n_mf = mf.NumberOfOperands
            print(f"[DIAG optimize] MFE 有 {n_mf} 个 operand, 优化前逐项:")
            MFEcol = ZOSAPI.Editors.MFE.MeritColumn
            for k in range(1, n_mf + 1):
                op_k = mf.GetOperandAt(k)
                try:
                    t = op_k.Type
                    # 从 cell 独立取四个参数重新求值，避开 op.Value 的缓存坑
                    try:
                        p1 = op_k.GetOperandCell(MFEcol.Param1).IntegerValue
                    except Exception:
                        p1 = 0
                    try:
                        p2 = op_k.GetOperandCell(MFEcol.Param2).IntegerValue
                    except Exception:
                        p2 = 0
                    try:
                        p3 = op_k.GetOperandCell(MFEcol.Param3).DoubleValue
                    except Exception:
                        p3 = 0.0
                    try:
                        p4 = op_k.GetOperandCell(MFEcol.Param4).DoubleValue
                    except Exception:
                        p4 = 0.0
                    v_cached = float(op_k.Value)
                    try:
                        v_fresh = float(mf.GetOperandValue(t, p1, p2, p3, p4, 0, 0, 0, 0))
                    except Exception:
                        v_fresh = None
                    print(f"  MFE[{k}] type={t} P=({p1},{p2},{p3:.3f},{p4:.3f}) "
                          f"target={op_k.Target} weight={op_k.Weight} "
                          f"cached={v_cached} fresh={v_fresh}")
                except Exception as e:
                    print(f"  MFE[{k}] 读取失败: {e}")
        except Exception as e:
            print(f"[DIAG optimize] MFE dump 失败: {e}")

        merit_before = float(opt.InitialMeritFunction)
        print(f"[DIAG optimize] InitialMeritFunction = {merit_before}")
        opt.RunAndWaitForCompletion()
        merit_after  = float(opt.CurrentMeritFunction)
        print(f"[DIAG optimize] CurrentMeritFunction (after) = {merit_after}")
        opt.Close()

        # ★ DLS完成后再跑一次QuickFocus，把像面拉到真正的焦点
        # 原因: MarginalRayHeight solve 在DLS迭代中可能被优化器改掉
        # QuickFocus 直接计算最佳焦点位置，比 solve 更可靠
        try:
            qf2 = sys_.Tools.OpenQuickFocus()
            if qf2 is not None:
                qf2.Criterion = ZOSAPI.Tools.General.QuickFocusCriterion.SpotSizeRadial
                qf2.UseCentroid = True
                qf2.RunAndWaitForCompletion()
                qf2.Close()
                print(f"[DIAG optimize] ✓ 优化后 QuickFocus 完成（像面已对准焦点）")
        except Exception as e:
            print(f"[DIAG optimize] ⚠ 优化后 QuickFocus 失败: {e}")
        # ★ 稳定等待: optimizer 关闭后 Zemax 内部状态需要短暂同步，
        #   立即发下一个请求会导致 RemoteDisconnected
        import time as _t; _t.sleep(1.0)

        # ★ P10 关键修复: 读取优化后的完整面型返回给 agent
        # 之前 bridge 只返回 merit/指标, 面型留在 Zemax 内存里没传回 —— agent 本地
        # lens['surfaces'] 还是老 radius, 下次 rms_calculator 读到老值会严重误判
        # (之前就是这样以为"RMS 0.018 达标", 其实 Zemax 真值 0.22).
        surfaces_after = []
        try:
            # 索引 0 是 object, NumberOfSurfaces-1 是 image, 中间是实物面
            for i in range(lde.NumberOfSurfaces - 1):
                s = lde.GetSurfaceAt(i)
                try:
                    r = float(s.Radius)
                    # ∞ / 极大值 归一为 0 (上游 agent 用 0 表示平面)
                    if abs(r) > 1e9:
                        r = 0.0
                except Exception:
                    r = 0.0
                try:
                    t = float(s.Thickness)
                    if abs(t) > 1e9:
                        t = 1.0e6   # object thickness 归为大数
                    # ★ 过滤负厚度：优化后回传给 agent 的面型不应含负厚度
                    # 否则下次 load_lens 会带着负厚度写入，系统从一开始就非法
                    elif t < 0 and i > 0:
                        mat_check = str(s.Material or "").upper()
                        t = 1.0 if mat_check not in ("", "AIR") else 0.5
                        print(f"[FIX-T-OUT] surf[{i}] 负厚度回传被修正为 {t}")
                except Exception:
                    t = 0.0
                try:
                    mat = str(s.Material or "")
                except Exception:
                    mat = ""
                try:
                    sd = float(s.SemiDiameter or 0.0)
                except Exception:
                    sd = 0.0
                surfaces_after.append({
                    "surface_num": float(i),
                    "radius":      r,
                    "thickness":   t,
                    "material":    mat or "AIR",
                    "semi_diameter": sd,
                })
            print(f"[DIAG optimize] 回传 {len(surfaces_after)} 面的优化后面型 (不含 image)")
        except Exception as e:
            print(f"[DIAG optimize] ⚠ 读取优化后面型失败: {e}")
            surfaces_after = []

        # 回传 merit 变化 + 真值指标（P3）+ 优化后面型（P10）
        m = _read_system_metrics(sys_)
        m.update({
            "merit":        round(merit_after, 6),      # alias，让 agent 端 data["merit"] 能用
            "merit_before": round(merit_before, 6),
            "merit_after":  round(merit_after, 6),
            "merit_delta":  round(merit_before - merit_after, 6),
            "cycles_used":  closest,
            "variables":    var_count,
            "surfaces_after": surfaces_after,    # ★ P10: 让 agent 能回写面型
        })

        # ★ 物理合理性检查：把异常显式写进返回，让轨迹可见供 self_evolve 蒸馏
        physics_warnings = []

        dist = m.get("distortion_per_field_pct", [])
        if len(dist) >= 2:
            # 现在用 REAY/理想像高计算，轴上(dist[0])应为0
            if abs(dist[0]) > 0.5:
                physics_warnings.append(
                    "⚠ [畸变异常] 轴上视场畸变=%.3f%%(应为0%%)，"
                    "主光线定义要求轴上畸变=0，"
                    "可能是REAY追迹失败或EFFL未准确对齐" % dist[0])
            # 非单调检查（只在畸变绝对值>1%时才报，避免数值噪音）
            if len(dist) >= 3 and max(abs(d) for d in dist) > 1.0:
                monotone = (dist[0] <= dist[1] <= dist[2] or
                            dist[0] >= dist[1] >= dist[2])
                if not monotone:
                    physics_warnings.append(
                        "⚠ [畸变非单调] 畸变随视场非单调"
                        "(%.2f%% → %.2f%% → %.2f%%)，"
                        "像差校正不均匀" % (dist[0], dist[1], dist[2]))

        rms = m.get("rms_per_field_mm", [])
        if len(rms) >= 2:
            if len(set(round(r, 6) for r in rms)) == 1:
                physics_warnings.append(
                    "⚠ [RMS异常] 三视场RMS完全相同(%.4fmm)，"
                    "视场追迹未真正区分各视场，结果不可信" % rms[0])
            if rms[0] > rms[-1] * 1.5 and rms[-1] > 0:
                physics_warnings.append(
                    "⚠ [场曲反常] 轴上RMS(%.4fmm)远大于全视场RMS(%.4fmm)，"
                    "场曲方向异常" % (rms[0], rms[-1]))
            if rms[-1] > rms[0] * 10 and rms[0] > 0:
                physics_warnings.append(
                    "⚠ [严重场曲] 全视场RMS(%.4fmm)是轴上(%.4fmm)的%.0f倍，"
                    "Petzval场曲未被校正，继续DLS优化收益有限，"
                    "建议考虑调整负镜结构" % (rms[-1], rms[0], rms[-1]/rms[0]))

        delta = m.get("merit_delta", 0)
        if abs(delta) < 0.001:
            physics_warnings.append(
                "⚠ [优化停滞] merit变化量=%.6f，优化器未找到下降方向，"
                "当前结构已到局部最优，继续优化无意义" % delta)

        if physics_warnings:
            m["physics_warnings"] = physics_warnings
            print("[zemax_optimize] 物理合理性检查:", file=sys.stderr, flush=True)
            for w in physics_warnings:
                print(f"  {w}", file=sys.stderr, flush=True)

        return jsonify(m)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════
#   /save_zmx  —— 把当前系统保存为 .zmx 文件
# ═════════════════════════════════════════════════════════════
@app.route("/save_zmx", methods=["POST"])
def save_zmx():
    try:
        payload   = request.json or {}
        save_path = payload.get("save_path", r"C:\zemax_results\result.zmx")
        lens_idx  = payload.get("lens_idx", 0)
        sys_ = _get_system()

        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        sys_.SaveAs(save_path)
        print(f"[save_zmx] ✓ 已保存到 {save_path}", flush=True)
        return jsonify({"ok": True, "path": save_path, "lens_idx": lens_idx})
    except Exception as e:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════
#   /layout  —— 用 ZOS-API CrossSectionExport 让 Zemax 自己渲染
#              直接出带真实光线的标准 Layout 图，和 GUI 一致
# ═════════════════════════════════════════════════════════════
@app.route("/layout", methods=["POST"])
def layout():
    try:
        sys_ = _get_system()

        # Try multiple save paths in case of permission issues
        for _bmp_path in [
            r"C:\Users\Right\zemax_layout_tmp.bmp",
            r"C:\zemax_layout_tmp.bmp",
            os.path.join(os.environ.get("TEMP", r"C:\Temp"), "zemax_layout_tmp.bmp"),
        ]:
            try:
                os.makedirs(os.path.dirname(_bmp_path) or ".", exist_ok=True)
                open(_bmp_path, "ab").close()   # test write permission
                save_path = _bmp_path
                break
            except Exception:
                continue
        else:
            save_path = r"C:\Users\Right\zemax_layout_tmp.bmp"
        print(f"[layout] save_path={save_path}")

        layout_tool = sys_.Tools.Layouts.OpenCrossSectionExport()
        layout_tool.OutputFileName = save_path
        layout_tool.SaveImageAsFile = True
        layout_tool.OutputPixelWidth = 1920
        layout_tool.OutputPixelHeight = 1080

        try:
            layout_tool.StartSurface = 1
            n_surf = sys_.LDE.NumberOfSurfaces
            layout_tool.EndSurface = n_surf - 1
            print(f"[layout] StartSurface=1 EndSurface={n_surf-1}")
        except Exception as _e:
            print(f"[layout] StartSurface/EndSurface failed: {_e}")

        try:
            layout_tool.NumberOfRays = 3
            print(f"[layout] NumberOfRays=3 OK")
        except Exception as _e:
            print(f"[layout] NumberOfRays failed: {_e}")

        print("[layout] RunAndWaitForCompletion ...")
        layout_tool.RunAndWaitForCompletion()
        layout_tool.Close()
        print(f"[layout] done, file exists={os.path.exists(save_path)}")

        if not os.path.exists(save_path):
            return jsonify({"error": "layout image not generated at " + save_path}), 500

        # BMP -> PNG
        try:
            from PIL import Image as _PIL_Image
            img = _PIL_Image.open(save_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            print(f"[layout] PNG ready, size={buf.getbuffer().nbytes}")
            return send_file(buf, mimetype="image/png")
        except ImportError:
            # PIL not installed: return BMP directly
            print("[layout] PIL not found, returning BMP")
            return send_file(save_path, mimetype="image/bmp")

    except Exception as e:
        tb = traceback.format_exc()
        print("[layout] ERROR:", tb)        # print full tb to Windows console
        return jsonify({"error": tb}), 500


# ═════════════════════════════════════════════════════════════
#   /spot_diagram  —— 点列图（Spot Diagram）
#   对每个视场各追一批光线到像面，把 X/Y 落点画成散点图。
#   使用 MFE 的 REAX / REAY operand 直接求像面坐标，比 BatchRayTrace
#   解析元组更稳定（不受版本字段顺序差异影响）。
# ═════════════════════════════════════════════════════════════
@app.route("/spot_diagram", methods=["POST", "GET"])
def spot_diagram():
    try:
        sys_ = _get_system()
        lde  = sys_.LDE
        img_surf = int(lde.NumberOfSurfaces - 1)   # image surface index (0-based)

        OpType = ZOSAPI.Editors.MFE.MeritOperandType
        mfe    = sys_.MFE

        # ── 从系统读视场数量，构造归一化 Hy 列表 ──────────────────────
        try:
            fields = sys_.SystemData.Fields
            nf     = fields.NumberOfFields
            y_max  = max((abs(float(fields.GetField(fi).Y)) for fi in range(1, nf + 1)),
                         default=1.0) or 1.0
            hy_vals = []
            for fi in range(1, nf + 1):
                try:
                    hy_vals.append(float(fields.GetField(fi).Y) / y_max)
                except Exception:
                    pass
        except Exception:
            hy_vals = [0.0, 0.7, 1.0]   # fallback

        # ── 光瞳采样网格（极坐标同心圆，共 ~61 个点）──────────────────
        import math
        pupil_pts = [(0.0, 0.0)]
        for ring_r, n_pts in [(0.25, 6), (0.5, 8), (0.75, 10), (1.0, 12)]:
            for k in range(n_pts):
                ang = 2 * math.pi * k / n_pts
                pupil_pts.append((ring_r * math.cos(ang), ring_r * math.sin(ang)))

        colors = ["#51cf66", "#339af0", "#ff6b6b",
                  "#ffd43b", "#cc5de8", "#ff922b"]

        # ── 为每个视场追光，收集 (x, y) 像面坐标 ──────────────────────
        field_spots = []
        for fi_idx, hy in enumerate(hy_vals):
            xs, ys = [], []
            for px, py in pupil_pts:
                try:
                    x_val = float(mfe.GetOperandValue(
                        OpType.REAX, img_surf, 0, 0.0, hy, px, py, 0.0, 0.0))
                    y_val = float(mfe.GetOperandValue(
                        OpType.REAY, img_surf, 0, 0.0, hy, px, py, 0.0, 0.0))
                    # 过滤追迹失败的点（返回 1e10 哨兵）
                    if abs(x_val) < 1e5 and abs(y_val) < 1e5:
                        xs.append(x_val)
                        ys.append(y_val)
                except Exception:
                    pass
            field_spots.append((hy, xs, ys))

        # ── 绘图：每个视场一个子图 ─────────────────────────────────────
        n_fields = len(field_spots)
        fig, axes = plt.subplots(1, n_fields,
                                 figsize=(3.5 * n_fields, 4.0),
                                 squeeze=False)
        fig.patch.set_facecolor("#0d1117")

        for col_idx, (hy, xs, ys) in enumerate(field_spots):
            ax = axes[0][col_idx]
            ax.set_facecolor("#0d1117")
            color = colors[col_idx % len(colors)]

            if xs and ys:
                # 以质心为原点（去除主光线偏移）
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                xs_c = [x - cx for x in xs]
                ys_c = [y - cy for y in ys]

                # ★ 离群点剔除：先算初步 RMS，去掉距中心 > 5×RMS 的光线
                # 原因：全反射/渐晕光线返回的坐标会通过 1e5 过滤但仍远大于正常值，
                #       混入后 RMS 被严重高估（实测从 ~30µm 膨胀到 ~500µm）
                if len(xs_c) >= 4:
                    rms_prelim = math.sqrt(sum(x**2 + y**2 for x, y in zip(xs_c, ys_c)) / len(xs_c))
                    threshold = max(rms_prelim * 5, 0.001)  # 至少 1µm 阈值
                    pairs = [(x, y) for x, y in zip(xs_c, ys_c) if math.hypot(x, y) <= threshold]
                    if len(pairs) >= 4:
                        xs_c = [p[0] for p in pairs]
                        ys_c = [p[1] for p in pairs]

                # Airy disk 半径: r_airy ≈ 1.22 × λ × F#
                # 用 F d 线 587.6 nm，从系统读 WFNO
                try:
                    fnum_val = float(mfe.GetOperandValue(
                        OpType.WFNO, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    if fnum_val is None or fnum_val > 1000:
                        raise ValueError
                except Exception:
                    fnum_val = 2.8
                lambda_d_mm = 0.0005876   # 587.6 nm in mm
                r_airy = 1.22 * lambda_d_mm * fnum_val

                # 散点
                ax.scatter(xs_c, ys_c, s=12, c=color, alpha=0.75, linewidths=0)
                # Airy disk 圆
                theta = [2 * math.pi * k / 120 for k in range(121)]
                ax.plot([r_airy * math.cos(t) for t in theta],
                        [r_airy * math.sin(t) for t in theta],
                        color="white", lw=0.8, alpha=0.5, linestyle="--",
                        label=f"Airy r={r_airy*1000:.1f} µm")

                # RMS 计算
                rms = math.sqrt(sum(x**2 + y**2 for x, y in zip(xs_c, ys_c)) / len(xs_c))
                ax.set_title(f"Field {hy:+.2f}\nRMS={rms*1000:.1f} µm",
                             color="white", fontsize=9)
                ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                          loc="upper right")
                # 轴范围：max(rms*4, airy*3) 保证可读
                half = max(rms * 4, r_airy * 3, 0.001)
                ax.set_xlim(-half, half)
                ax.set_ylim(-half, half)
            else:
                ax.set_title(f"Field {hy:+.2f}\n(no rays)", color="white", fontsize=9)

            ax.set_aspect("equal")
            ax.set_xlabel("X (mm)", color="white", fontsize=8)
            ax.set_ylabel("Y (mm)", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            ax.axhline(0, color="#444", lw=0.5)
            ax.axvline(0, color="#444", lw=0.5)

        fig.suptitle("Spot Diagram", color="white", fontsize=11, y=1.01)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _init()
    # threaded=True: 每个请求用独立线程，避免连续调用时连接被单线程队列阻塞
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
