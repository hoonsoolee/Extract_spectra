"""
ceres_demux.py — CBDF 역다중화 파서 v2 (세그먼트 분할 지원)
============================================================
v2 변경: frame_no 갭 기준 세그먼트(A, B, C...) 자동 분할
  → 공식 Export2Bil의 .A. .B. 출력 방식과 동일

레코드 구조 (실측 확정):
  [CRC32 4B][payload_size u64 LE][type 6B][payload]
  이미지 payload = [서브헤더 43B][픽셀 uint16 LE (bands, samples)]
  서브헤더: 01 05 02 | frame_no u16 | 6B | ts_ns u64 | bands u64 | samples u64 | n_pix u64
  타입: 040902001a00=VNIR, 030902001a00=SWIR, 020803000f00=GPS/IMU

사용:
  python ceres_demux.py probe   FILE.ceres
  python ceres_demux.py convert FILE.ceres -o OUT [--segment A] [--gap 30]
  python ceres_demux.py verify  FILE.ceres --bil-dir extract-bil [--gap 30]
  python ceres_demux.py gps     FILE.ceres -o gps_imu.csv
"""
import os, re, sys, json, struct, argparse, string, datetime
from pathlib import Path
import numpy as np

GLOBAL_HEADER = 65536
REC_HDR, SUBHDR = 18, 43
TYPE_VNIR = bytes.fromhex('040902001a00')
TYPE_SWIR = bytes.fromhex('030902001a00')
TYPE_GPS  = bytes.fromhex('020803000f00')
IMG_TYPES = {TYPE_VNIR: 'VNIR', TYPE_SWIR: 'SWIR'}
MAX_REC = 512 * 1024 * 1024


def resync(f, pos, size):
    CH, tail = 16*1024*1024, b''
    while pos < size:
        f.seek(pos); chunk = f.read(min(CH, size-pos))
        if not chunk: return -1
        buf = tail + chunk
        hits = [i for t in IMG_TYPES for i in [buf.find(t)] if i != -1]
        if hits:
            cand = pos - len(tail) + min(hits) - 12
            if cand >= 0: return cand
        tail = buf[-6:]; pos += len(chunk)
    return -1


def walk_records(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        pos, n = GLOBAL_HEADER, 0
        while pos + REC_HDR <= size:
            f.seek(pos); h = f.read(REC_HDR)
            if len(h) < REC_HDR: break
            psize = int.from_bytes(h[4:12], 'little'); typ = h[12:18]
            if psize == 0 or psize > MAX_REC or pos + REC_HDR + psize > size:
                pos = resync(f, pos+1, size)
                if pos < 0: break
                continue
            yield pos, typ, psize
            pos += REC_HDR + psize; n += 1
            if n % 20000 == 0:
                print(f"  ... {pos/1e9:.2f} GB ({n:,} rec)", file=sys.stderr)


def parse_sub(f, off):
    f.seek(off + REC_HDR); sh = f.read(SUBHDR)
    if len(sh) < SUBHDR or sh[:3] != b'\x01\x05\x02': return None
    return {'frame_no': int.from_bytes(sh[3:5],'little'),
            'ts_ns':    int.from_bytes(sh[11:19],'little'),
            'bands':    int.from_bytes(sh[19:27],'little'),
            'samples':  int.from_bytes(sh[27:35],'little'),
            'n_pixels': int.from_bytes(sh[35:43],'little'),
            'pix_off':  off + REC_HDR + SUBHDR}


def collect_frames(path):
    frames, others = {'VNIR': [], 'SWIR': []}, {}
    with open(path, 'rb') as f:
        for off, typ, psize in walk_records(path):
            name = IMG_TYPES.get(typ)
            if name:
                info = parse_sub(f, off)
                if info and info['n_pixels']*2 == psize - SUBHDR:
                    info['bytes'] = psize - SUBHDR
                    frames[name].append(info); continue
            others[typ.hex()] = others.get(typ.hex(), 0) + 1
    for name in frames:
        fr = frames[name]
        if fr:
            nos = [x['frame_no'] for x in fr]
            fr.sort(key=(lambda x: x['ts_ns']) if max(nos)-min(nos) > 60000
                    else (lambda x: x['frame_no']))
    return frames, others


def split_segments(fr, gap_thr=30):
    """frame_no 연속 구간으로 분할 → [(letter, [frames]), ...]"""
    if not fr: return []
    segs, cur = [], [fr[0]]
    for a, b in zip(fr, fr[1:]):
        if b['frame_no'] - a['frame_no'] > gap_thr:
            segs.append(cur); cur = [b]
        else:
            cur.append(b)
    segs.append(cur)
    letters = list(string.ascii_uppercase)
    return [(letters[i] if i < 26 else f'S{i}', s) for i, s in enumerate(segs)]


def wavelengths_from(base, name, n_bands):
    j = base / f'wavelengths_{name.lower()}.json'
    if j.exists():
        wl = json.loads(j.read_text())
        if len(wl) == n_bands: return wl
    for d in (base/'extract-bil', base/'extract_bil', base):
        if d.is_dir():
            for h in d.glob(f'*{name.lower()}*.hdr'):
                m = re.search(r'wavelength\s*=\s*\{([^}]*)\}',
                              h.read_text(errors='ignore'), re.S|re.I)
                if m:
                    wl = [float(x) for x in m.group(1).replace('\n',' ').split(',') if x.strip()]
                    if len(wl) == n_bands: return wl
    rng = (404.895, 996.597) if name=='VNIR' else (953.756, 2514.320)
    return list(np.linspace(*rng, n_bands))


def write_hdr(path, lines, samples, bands, wl):
    wl_s = ', '.join(f'{w:.4f}' for w in wl)
    Path(path).write_text(
        f"ENVI\ninterleave = bil\nheader offset = 0\nfile type = ENVI Standard\n"
        f"data type = 12\nbyte order = 0\nbands = {bands}\nlines = {lines}\n"
        f"samples = {samples}\nwavelength units = nm\nwavelength = {{{wl_s}}}\n")


def cmd_probe(path, gap):
    print(f"\n{'='*62}\n  {os.path.basename(path)} "
          f"({os.path.getsize(path)/1e9:.2f} GB)\n{'='*62}")
    frames, others = collect_frames(path)
    for name, fr in frames.items():
        if not fr: continue
        b, s = fr[0]['bands'], fr[0]['samples']
        print(f"\n  {name}: 총 {len(fr)} 프레임 ({b} bands × {s} samples)")
        for letter, seg in split_segments(fr, gap):
            t0 = datetime.datetime.fromtimestamp(seg[0]['ts_ns']/1e9)
            dur = (seg[-1]['ts_ns'] - seg[0]['ts_ns'])/1e9
            nos = [x['frame_no'] for x in seg]
            drop = max(nos)-min(nos)+1-len(seg)
            print(f"    [{letter}] {len(seg):5d} lines  frame {min(nos)}~{max(nos)}"
                  f"  {dur:6.1f}s  시작 {t0:%H:%M:%S}"
                  f"{'  (내부드롭 '+str(drop)+')' if drop else ''}")
    print(f"\n  기타 레코드: " + ', '.join(f"{k}×{v:,}" for k, v
          in sorted(others.items(), key=lambda x:-x[1])[:8]) + "\n")


def cmd_convert(path, out_dir, gap, only_seg=None):
    base = Path(path).resolve().parent
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(path).stem
    frames, _ = collect_frames(path)
    with open(path, 'rb') as f:
        for name, fr in frames.items():
            if not fr: continue
            b, s = fr[0]['bands'], fr[0]['samples']
            wl = wavelengths_from(base, name, b)
            for letter, seg in split_segments(fr, gap):
                if only_seg and letter != only_seg.upper(): continue
                out = str(Path(out_dir) / f"{stem}.{letter}.{name.lower()}.bil")
                with open(out, 'wb') as fo:
                    for info in seg:
                        f.seek(info['pix_off']); fo.write(f.read(info['bytes']))
                write_hdr(out[:-4]+'.hdr', len(seg), s, b, wl)
                print(f"  ✓ [{letter}] {name}: {len(seg)} lines → {out}")
    print("  변환 완료!")


def cmd_verify(path, bil_dir, gap):
    d = Path(bil_dir)
    frames, _ = collect_frames(path)
    with open(path, 'rb') as f:
        for name, fr in frames.items():
            if not fr: continue
            for letter, seg in split_segments(fr, gap):
                # 공식 파일: 이름에 .{letter}. 과 센서명 포함
                offi = next(iter(sorted(d.glob(f'*.{letter}.{name.lower()}*.bil'))), None) \
                    or next(iter(sorted(d.glob(f'*{letter}_{name.lower()}*.bil'))), None)
                if not offi:
                    print(f"  [{letter}] {name}: 공식 BIL 없음 (라인 {len(seg)}) — 스킵")
                    continue
                line_b = seg[0]['bytes']
                off_lines = offi.stat().st_size // line_b
                n = min(len(seg), off_lines)
                ok, bad = 0, []
                with open(offi, 'rb') as fb:
                    for i in range(n):
                        f.seek(seg[i]['pix_off']); a = f.read(line_b)
                        fb.seek(i*line_b); bb = fb.read(line_b)
                        if a == bb: ok += 1
                        else:
                            bad.append(i)
                            if len(bad) >= 5: break
                extra = '' if len(seg)==off_lines else f"  (파서 {len(seg)} vs 공식 {off_lines} lines)"
                print(f"  [{letter}] {name}: {ok}/{n} 일치 "
                      f"{'✓ 완전 일치!!' if not bad else f'✗ {bad}'}{extra}")


def cmd_gps(path, out_csv):
    rows = []
    with open(path, 'rb') as f:
        for off, typ, psize in walk_records(path):
            if typ == TYPE_GPS and psize == 49:
                f.seek(off + REC_HDR); p = f.read(49)
                ts = [int.from_bytes(p[1+i*8:9+i*8],'little') for i in range(3)]
                fl = struct.unpack_from('<6f', p, 25)
                rows.append(tuple(ts)+fl)
    with open(out_csv, 'w') as fo:
        fo.write("t1,t2,t3,f1,f2,f3,f4,f5,f6\n")
        for r in rows: fo.write(','.join(str(x) for x in r)+'\n')
    print(f"  GPS/IMU {len(rows):,}개 → {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('probe','convert','verify'):
        p = sub.add_parser(c); p.add_argument('file')
        p.add_argument('--gap', type=int, default=30,
                       help='세그먼트 분할 frame_no 갭 임계 (기본 30)')
        if c=='convert':
            p.add_argument('-o','--out', default='./demux_out')
            p.add_argument('--segment', help='특정 세그먼트만 (예: A)')
        if c=='verify':
            p.add_argument('--bil-dir', default='extract-bil')
    p4 = sub.add_parser('gps'); p4.add_argument('file')
    p4.add_argument('-o','--out', default='gps_imu.csv')
    a = ap.parse_args()
    if a.cmd=='probe':   cmd_probe(a.file, a.gap)
    elif a.cmd=='convert': cmd_convert(a.file, a.out, a.gap, a.segment)
    elif a.cmd=='verify':  cmd_verify(a.file, a.bil_dir, a.gap)
    elif a.cmd=='gps':     cmd_gps(a.file, a.out)


if __name__ == '__main__':
    main()
