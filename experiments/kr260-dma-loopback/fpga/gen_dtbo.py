#!/usr/bin/env python3
# gen_dtbo.py — STAGE B only. KR260 dtbo via the Vitis 2025.2 Python flow,
# the replacement for the removed xsct/createdts. Produces the full,
# vendor-correct PL overlay (with a zocl node injected via user_dtsi) so XRT
# exposes an accelerator device.
#
# Run on the Windows VM (cmd):
#   C:\AMDDesignTools\2025.2.1\Vitis\bin\vitis.bat -s fpga\gen_dtbo.py
#
# Adapted from Vitis\cli\examples\embedded\dtb_support.py. Key KR260 changes:
#   * cpu = psu_cortexa53   (ZynqMP; the example's psv_cortexa72 is Versal)
#   * dt_overlay = "1"      (emit a .dtbo overlay, not a full base .dtb)
#   * user_dtsi = zocl.dtsi (merge our zocl node — Stage B needs the XRT device)
#
# Stage A does NOT use this file (it uses overlay/loopback.dts + dtc on-board).

import os
import shutil
import vitis

HERE = os.path.dirname(os.path.abspath(__file__))
XSA       = os.path.join(HERE, "out", "loopback.xsa")
ZOCL_DTSI = os.path.join(HERE, "..", "overlay", "zocl.dtsi")  # created in Stage B
WS        = r"C:\tmp\kr260_ws"

client = vitis.create_client()
if os.path.isdir(WS):
    shutil.rmtree(WS, ignore_errors=True)
client.set_workspace(WS)

# Advanced options: dt_overlay=1 -> .dtbo; user_dtsi -> merge our zocl fragment.
# Other supported keys: sdt_repo (pin device-tree-xlnx), board_dtsi.
adv = client.create_advanced_options_dict(
    dt_overlay="1",
    user_dtsi=ZOCL_DTSI,
)

platform = client.create_platform_component(
    name             = "kr260_pfm",
    hw_design        = XSA,
    os               = "linux",
    cpu              = "psu_cortexa53",
    domain_name      = "linux_psu_cortexa53",
    generate_dtb     = True,
    advanced_options = adv,
)
platform.build()
platform.get_domain("linux_psu_cortexa53").recompile_dtb()

vitis.dispose()
print(f"\nDone. Search for the generated *.dtbo under: {WS}")
print("Copy it to the board as penzai-loopback.dtbo and load via deploy.sh.")
