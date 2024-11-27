# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ["src/gui.py"],
    pathex=[],
    binaries=[],
    datas=[("src/submodules/BirdNET-Analyzer", "submodules/BirdNET-Analyzer")],
    hiddenimports=["tensorflow","resampy","librosa"],
    hookspath=["src"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui',
)
