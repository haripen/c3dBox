#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Patch C3D Labels UI — aliasing + condense + reconcile

import os, sys, re, json, traceback
from pathlib import Path
from collections import Counter, defaultdict

import ezc3d  # REQUIRED
from PyQt5 import QtWidgets, QtCore

def emg_channel_key(label: str):
    s = str(label).lower()
    if 'emg' not in s:
        return None
    m = re.search(r'emg[\.\s_\-]*([0-9]{1,2})', s)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return None
        if 1 <= n <= 99:
            return f'emg#{n}'
    return None

def base_group_key(label: str) -> str:
    return re.sub(r'[^0-9a-zA-Z]', '', str(label)).lower()

def group_key(label: str) -> str:
    ek = emg_channel_key(label)
    return ek if ek else base_group_key(label)

def _to_int(v):
    try:
        import numpy as np  # noqa: F401
        if hasattr(v, 'shape'):
            try:
                return int(v.item())
            except Exception:
                return int(v.reshape(-1)[0])
        if isinstance(v, (list, tuple)):
            return int(v[0])
        return int(v)
    except Exception:
        return int(list(v)[0])

def _list_blocks(section, base='LABELS'):
    keys = []
    i = 1
    while True:
        key = base if i == 1 else f'{base}{i}'
        if key not in section:
            break
        keys.append(key)
        i += 1
    return keys

def _lens(section, base='LABELS'):
    return [(k, len(section[k]['value'])) for k in _list_blocks(section, base=base)]

def _total_len(section, base='LABELS'):
    return sum(len(section[k]['value']) for k in _list_blocks(section, base=base))

def collect_param_labels(section) -> list:
    labels = []
    for key in _list_blocks(section, base='LABELS'):
        vals = section[key]['value']
        seq = list(vals) if hasattr(vals, '__iter__') else [vals]
        for v in seq:
            s = '' if v is None else str(v).strip()
            if s:
                labels.append(s)
    return labels

class LabelInventory:
    def __init__(self):
        self.point_groups = defaultdict(Counter)
        self.analog_groups = defaultdict(Counter)

    def add_points(self, labels):
        for lab in labels:
            if lab.startswith('*'):
                continue
            self.point_groups[group_key(lab)][lab] += 1

    def add_analogs(self, labels):
        for lab in labels:
            self.analog_groups[group_key(lab)][lab] += 1

    def ingest_c3d(self, path: str):
        c3d = ezc3d.c3d(path)
        pts = collect_param_labels(c3d['parameters']['POINT'])
        ans = collect_param_labels(c3d['parameters']['ANALOG'])
        self.add_points(pts)
        self.add_analogs(ans)

    def most_frequent_mapping_and_mutable_sets(self):
        pm, am = {}, {}
        pmut, amut = set(), set()
        for gk, counter in self.point_groups.items():
            if counter:
                pm[gk] = counter.most_common(1)[0][0]
                if len(counter) >= 2:
                    pmut.add(gk)
        for gk, counter in self.analog_groups.items():
            if counter:
                am[gk] = counter.most_common(1)[0][0]
                if len(counter) >= 2:
                    amut.add(gk)
        return pm, am, pmut, amut

def condense_point_blocks(c3d, debug=False, log_cb=lambda m: None):
    # Merge POINT LABELS blocks and DESCRIPTIONS blocks into single blocks if total labels <= 255
    psec = c3d['parameters']['POINT']
    label_keys = _list_blocks(psec, 'LABELS')
    total_labels = _total_len(psec, 'LABELS')
    if total_labels > 255 or len(label_keys) <= 1:
        if debug:
            log_cb(f"[condense] skip (total_labels={total_labels}, blocks={len(label_keys)})")
        return 0, 0

    merged_labels = []
    for k in label_keys:
        merged_labels.extend(list(psec[k]['value']))

    desc_keys = _list_blocks(psec, 'DESCRIPTIONS')
    merged_desc = []
    for k in desc_keys:
        merged_desc.extend(list(psec[k]['value']))

    psec['LABELS']['value'] = list(merged_labels)
    for k in label_keys[1:]:
        del psec[k]
        if debug:
            log_cb(f"[condense] removed POINT/{k}")

    trimmed_desc = 0
    if desc_keys:
        if len(merged_desc) > len(merged_labels):
            trimmed_desc = len(merged_desc) - len(merged_labels)
            merged_desc = merged_desc[:len(merged_labels)]
            if debug:
                log_cb(f"[condense] trimmed DESCRIPTIONS tail by {trimmed_desc} to match labels")
        if 'DESCRIPTIONS' in psec:
            psec['DESCRIPTIONS']['value'] = list(merged_desc)
        for k in desc_keys[1:]:
            if k in psec:
                del psec[k]
                if debug:
                    log_cb(f"[condense] removed POINT/{k}")

    if debug:
        lens_lbl = _lens(psec, 'LABELS')
        lens_desc = _lens(psec, 'DESCRIPTIONS')
        log_cb(f"[condense] RESULT: LABELS={lens_lbl}, DESCRIPTIONS={lens_desc}")
    return 1, trimmed_desc

def reconcile_point_label_counts(c3d, debug=False, log_cb=lambda m: None):
    # Trim excess POINT labels (mirror DESCRIPTIONS) so total labels == nPoints; set POINT.USED=nPoints
    psec = c3d['parameters']['POINT']
    labels_keys = _list_blocks(psec, base='LABELS')
    nb_points = c3d['data']['points'].shape[1]
    total_labels = _total_len(psec, base='LABELS')
    used_before = _to_int(psec['USED']['value']) if 'USED' in psec else None

    if debug:
        log_cb(f"[reconcile] nPoints={nb_points}, USED={used_before}, LABELS*={_lens(psec,'LABELS')}, DESCRIPTIONS*={_lens(psec,'DESCRIPTIONS')}")

    if total_labels == nb_points:
        if 'USED' in psec and used_before != nb_points:
            psec['USED']['value'] = [nb_points]
            if debug:
                log_cb(f"[reconcile] set POINT.USED -> {nb_points}")
        return 0

    if total_labels < nb_points:
        raise ValueError(f"[reconcile] Fewer POINT labels ({total_labels}) than data points ({nb_points}); not adding labels.")

    over = total_labels - nb_points
    removed = 0
    for k in reversed(labels_keys):
        if over <= 0:
            break
        block = psec[k]['value']
        take = min(len(block), over)
        if take > 0:
            if debug:
                log_cb(f"[reconcile] trimming {take} from POINT/{k} tail (len={len(block)})")
            psec[k]['value'] = list(block[:-take])
            dk = k.replace('LABELS','DESCRIPTIONS')
            if dk in psec:
                dblock = psec[dk]['value']
                if len(dblock) >= take:
                    psec[dk]['value'] = list(dblock[:-take])
                    if debug:
                        log_cb(f"[reconcile] also trimmed DESCRIPTIONS in {dk} by {take}")
            removed += take
            over -= take

    total_after = _total_len(psec, base='LABELS')
    if total_after != nb_points:
        raise RuntimeError(f"[reconcile] Post-trim mismatch: total labels={total_after}, nPoints={nb_points}")

    if 'USED' in psec:
        psec['USED']['value'] = [nb_points]
        if debug:
            log_cb(f"[reconcile] set POINT.USED -> {nb_points}")

    if debug:
        log_cb(f"[reconcile] RESULT: LABELS*={_lens(psec,'LABELS')}, DESCRIPTIONS*={_lens(psec,'DESCRIPTIONS')} (removed {removed})")
    return removed

def _patch_label_section(c3d, section_name: str, mapping: dict, mutable_groups: set, log_cb, debug=False):
    sec = c3d['parameters'][section_name]
    total_changed = 0
    for key in _list_blocks(sec, base='LABELS'):
        labels_ref = sec[key]['value']
        if debug:
            log_cb(f"--- {section_name}/{key} BEFORE (len={len(labels_ref)}): {list(labels_ref)}")
        changed_here = 0
        for idx in range(len(labels_ref)):
            s = str(labels_ref[idx])
            if not s or s.startswith('*'):
                continue
            gk = group_key(s)
            if gk in mutable_groups and gk in mapping:
                target = mapping[gk]
                if target != s:
                    labels_ref[idx] = target
                    changed_here += 1
                    if debug:
                        log_cb(f"{section_name}/{key}[{idx}]: '{s}' -> '{target}'")
        total_changed += changed_here
        if debug:
            log_cb(f"--- {section_name}/{key} AFTER  (len={len(labels_ref)}): {list(labels_ref)}")
            log_cb(f"--- {section_name}/{key} changed: {changed_here}")
    return total_changed

def _sanity_log_counts(c3d, log_cb):
    psec = c3d['parameters']['POINT']
    asec = c3d['parameters']['ANALOG']
    used_p = _to_int(psec['USED']['value']) if 'USED' in psec else None
    used_a = _to_int(asec['USED']['value']) if 'USED' in asec else None
    log_cb(f"[sanity] POINT.USED={used_p}, LABELS*={_lens(psec,'LABELS')}, DESCRIPTIONS*={_lens(psec,'DESCRIPTIONS')}")
    log_cb(f"[sanity] ANALOG.USED={used_a}, LABELS*={_lens(asec,'LABELS')}")

def patch_c3d_file(c3d_path: str, point_map: dict, analog_map: dict,
                   point_mutable: set, analog_mutable: set,
                   dry_run: bool=False, debug: bool=False,
                   condense: bool=False, reconcile: bool=False,
                   log_cb=lambda m: None):
    c3d = ezc3d.c3d(c3d_path)
    if debug:
        log_cb(f"==> Patching: {c3d_path} (dry_run={dry_run}, debug={debug}, condense={condense}, reconcile={reconcile})")
        _sanity_log_counts(c3d, log_cb)

    if condense:
        condense_point_blocks(c3d, debug=debug, log_cb=log_cb)

    if reconcile:
        reconcile_point_label_counts(c3d, debug=debug, log_cb=log_cb)

    changed_p = _patch_label_section(c3d, 'POINT', point_map, point_mutable, log_cb, debug=debug)
    changed_a = _patch_label_section(c3d, 'ANALOG', analog_map, analog_mutable, log_cb, debug=debug)

    if debug:
        log_cb(f"[summary] changed POINT={changed_p}, ANALOG={changed_a}")
        _sanity_log_counts(c3d, log_cb)

    if dry_run:
        log_cb(f"[dry] would write: {c3d_path}")
        return True

    if 'meta_points' in c3d['data']:
        del c3d['data']['meta_points']
        if debug:
            log_cb("[write] deleted data['meta_points'] to force rebuild")

    c3d.write(c3d_path)
    log_cb(f"patched: {c3d_path}")
    return True

class GroupTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(['Group', 'Total', 'Variants', 'Most Frequent (editable)'])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.SelectedClicked)

    def load_from_groups(self, groups: dict, default_mapping: dict):
        self.setRowCount(0)
        rows = []
        for gk, counter in groups.items():
            rows.append((gk, sum(counter.values()), counter))
        rows.sort(key=lambda t: (-t[1], t[0]))
        for gk, total, counter in rows:
            r = self.rowCount()
            self.insertRow(r)
            self.setItem(r, 0, QtWidgets.QTableWidgetItem(gk))
            self.setItem(r, 1, QtWidgets.QTableWidgetItem(str(total)))
            variants = ', '.join([f'{lab}({cnt})' for lab, cnt in counter.most_common(6)])
            item_var = QtWidgets.QTableWidgetItem(variants)
            item_var.setToolTip('\n'.join([f'{lab}: {cnt}' for lab, cnt in counter.most_common()]))
            self.setItem(r, 2, item_var)
            mf = default_mapping.get(gk, counter.most_common(1)[0][0] if counter else '')
            item_mf = QtWidgets.QTableWidgetItem(mf)
            item_mf.setFlags(item_mf.flags() | QtCore.Qt.ItemIsEditable)
            self.setItem(r, 3, item_mf)
        self.resizeColumnsToContents()

    def export_mapping(self) -> dict:
        out = {}
        for r in range(self.rowCount()):
            gk = self.item(r, 0).text().strip()
            mf = self.item(r, 3).text().strip()
            if gk and mf:
                out[gk] = mf
        return out

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Patch C3D Labels — Aliasing + Condense + Reconcile')
        self.resize(1280, 900)
        self.inventory = LabelInventory()
        self.point_default = {}
        self.analog_default = {}
        self.point_mutable = set()
        self.analog_mutable = set()
        self._init_ui()

    def _init_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        # Root selector
        h = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit()
        self.root_edit.setPlaceholderText('Select root folder (must exist)')
        b = QtWidgets.QPushButton('Browse…'); b.clicked.connect(self._browse_root)
        h.addWidget(QtWidgets.QLabel('Root:')); h.addWidget(self.root_edit); h.addWidget(b)
        v.addLayout(h)

        # Controls
        hb = QtWidgets.QHBoxLayout()
        self.btn_scan = QtWidgets.QPushButton('1) Build Label Inventory'); self.btn_scan.clicked.connect(self._scan)
        self.btn_patch = QtWidgets.QPushButton('4) Patch C3D Files'); self.btn_patch.clicked.connect(self._patch)
        self.btn_save_map = QtWidgets.QPushButton('2) Save Mapping JSON'); self.btn_save_map.clicked.connect(self._save_map)
        self.btn_load_map = QtWidgets.QPushButton('3) Load Mapping JSON'); self.btn_load_map.clicked.connect(self._load_map)
        self.chk_dry = QtWidgets.QCheckBox('Dry run'); self.chk_dry.setChecked(False)
        self.chk_debug = QtWidgets.QCheckBox('Debug'); self.chk_debug.setChecked(False)
        self.chk_condense = QtWidgets.QCheckBox('Condense POINT LABELS/DESCRIPTIONS (<=255)'); self.chk_condense.setChecked(True)
        self.chk_reconcile = QtWidgets.QCheckBox('Reconcile POINT labels to nPoints (trim excess)'); self.chk_reconcile.setChecked(False)

        hb.addWidget(self.btn_scan); hb.addStretch(1)
        hb.addWidget(self.btn_save_map); hb.addWidget(self.btn_load_map)
        hb.addStretch(1)
        hb.addWidget(self.chk_dry); hb.addWidget(self.chk_debug); hb.addWidget(self.chk_condense); hb.addWidget(self.chk_reconcile)
        hb.addWidget(self.btn_patch)
        v.addLayout(hb)

        # Tables
        self.tabs = QtWidgets.QTabWidget()
        self.table_point = GroupTable()
        self.table_analog = GroupTable()
        self.tabs.addTab(self.table_point, 'Point Labels')
        self.tabs.addTab(self.table_analog, 'Analog Labels')
        v.addWidget(self.tabs, 1)

        # Mapping JSON preview
        self.json_preview = QtWidgets.QPlainTextEdit()
        self.json_preview.setReadOnly(True)
        self.json_preview.setPlaceholderText('Loaded or saved mapping JSON will be shown here…')
        v.addWidget(QtWidgets.QLabel('Mapping JSON preview:'))
        v.addWidget(self.json_preview, 1)

        # Log
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet('background:#f7f7f7;')
        v.addWidget(QtWidgets.QLabel('Log:'))
        v.addWidget(self.log, 1)
        self.statusBar().showMessage('Ready')

    def _browse_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Root Folder')
        if d: self.root_edit.setText(d)

    def _find(self, root: str, exts):
        if not os.path.isdir(root):
            raise FileNotFoundError(f'Root folder does not exist: {root}')
        out=[]; exts=tuple(exts)
        for dp,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(exts):
                    out.append(str(Path(dp)/f))
        return out

    def _scan(self):
        root = self.root_edit.text().strip()
        if not root:
            raise FileNotFoundError('No root selected.')
        if not os.path.isdir(root):
            raise FileNotFoundError(f'Root folder does not exist: {root}')

        self.inventory = LabelInventory()
        c3ds = self._find(root, ['.c3d'])
        if not c3ds:
            raise FileNotFoundError('No C3D files found under the selected root.')

        self.log.clear()
        self._log(f'Scanning {len(c3ds)} C3D files (POINT/ANALOG LABELS)…')
        for i, p in enumerate(c3ds, 1):
            self.statusBar().showMessage(f'[C3D] {i}/{len(c3ds)}')
            try:
                self.inventory.ingest_c3d(p)
            except Exception:
                tb = traceback.format_exc()
                QtWidgets.QMessageBox.critical(self, 'C3D Read Error', f'File: {p}{tb}')
                raise

        self.point_default, self.analog_default, self.point_mutable, self.analog_mutable = (
            self.inventory.most_frequent_mapping_and_mutable_sets()
        )
        self.table_point.load_from_groups(self.inventory.point_groups, self.point_default)
        self.table_analog.load_from_groups(self.inventory.analog_groups, self.analog_default)
        self.statusBar().showMessage('Inventory built — edit “Most Frequent” cells, then Save/Load Mapping, then Patch.')

    def _cur_maps(self):
        pt_raw = self.table_point.export_mapping()
        an_raw = self.table_analog.export_mapping()
        return pt_raw, an_raw, self.point_mutable, self.analog_mutable

    def _save_map(self):
        pt_raw, an_raw, pmut, amut = self._cur_maps()
        out = {'point': {'group_to_mostfrequent_raw': pt_raw, 'mutable_groups': list(pmut)},
               'analog': {'group_to_mostfrequent_raw': an_raw, 'mutable_groups': list(amut)}}
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Mapping JSON', 'label_mapping.json', 'JSON (*.json)')
        if not path: return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        self.json_preview.setPlainText(json.dumps(out, indent=2, ensure_ascii=False))
        self._log(f'Saved mapping to {path}')

    def _load_map(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Mapping JSON', '', 'JSON (*.json)')
        if not path: return
        with open(path, 'r', encoding='utf-8') as f:
            data=json.load(f)
        self.json_preview.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
        pt_raw = data.get('point', {}).get('group_to_mostfrequent_raw', {})
        an_raw = data.get('analog', {}).get('group_to_mostfrequent_raw', {})
        self.point_mutable = set(data.get('point', {}).get('mutable_groups', []))
        self.analog_mutable = set(data.get('analog', {}).get('mutable_groups', []))
        if pt_raw: self.table_point.load_from_groups(self.inventory.point_groups, pt_raw)
        if an_raw: self.table_analog.load_from_groups(self.inventory.analog_groups, an_raw)
        self._log(f'Loaded mapping from {path}')

    def _patch(self):
        root = self.root_edit.text().strip()
        if not root or not os.path.isdir(root):
            raise FileNotFoundError('Root folder is invalid or not set.')
        pt_raw, an_raw, pmut, amut = self._cur_maps()
        if not pt_raw and not an_raw:
            raise RuntimeError('No mapping present. Build inventory or load a mapping first.')
        c3ds = self._find(root, ['.c3d'])
        if not c3ds:
            raise FileNotFoundError('No C3D files found under the selected root.')

        dry = self.chk_dry.isChecked()
        debug = self.chk_debug.isChecked()
        condense = self.chk_condense.isChecked()
        reconcile = self.chk_reconcile.isChecked()

        if not dry:
            reply = QtWidgets.QMessageBox.question(self, 'Confirm Patch', 'Overwrite C3D files in place?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes: return

        self._log(f'Patching {len(c3ds)} C3D files… (dry_run={dry}, debug={debug}, condense={condense}, reconcile={reconcile})')
        for i, p in enumerate(c3ds, 1):
            self.statusBar().showMessage(f'{i}/{len(c3ds)}: {p}')
            try:
                patch_c3d_file(p, pt_raw, an_raw, pmut, amut,
                               dry_run=dry, debug=debug,
                               condense=condense, reconcile=reconcile,
                               log_cb=self._log)
            except Exception:
                tb = traceback.format_exc()
                QtWidgets.QMessageBox.critical(self, 'Patch Error', f'File: {p}{tb}')
                raise
        self._log(f"Done. Processed {len(c3ds)} files.")
        self.statusBar().showMessage('Finished')

    def _log(self, m: str):
        self.log.append(m); print(m)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
