#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sort weight-start events by time from archivedTimeline.json")
    ap.add_argument("infile", nargs="?", default="archivedTimeline.json", help="Input JSON (default: archivedTimeline.json)")
    ap.add_argument("--out-json", default="weight_starts_sorted.json", help="Output JSON filename")
    ap.add_argument("--out-txt", default="weight_starts_sorted.txt", help="Output TXT filename")
    ap.add_argument("--max-per-time", type=int, default=0,
                    help="If >0, cap the number of printed entries per exact time in the TXT output")
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of layers.")

    events: List[Dict[str, Any]] = []

    # Pointwise condensation:
    # - out channels are processed in blocks of 32 => out_block = out_ch // 32
    # - input channels from prev layer are in blocks of 32 => w = prev_in_ch // 32
    # Group by (time, layer, out_block, w)
    pw_group: Dict[Tuple[int, int, int, int], int] = defaultdict(int)

    for layer_obj in data:
        lt = layer_obj.get("type")
        layer_idx = int(layer_obj.get("layer"))

        if lt == "Depthwise":
            for ch in layer_obj.get("channels", []):
                out_ch = int(ch.get("out_ch"))
                for t in as_list(ch.get("starts")):
                    if t is None:
                        continue
                    t = int(t)
                    events.append({
                        "time": t,
                        "kind": "DW",
                        "layer": layer_idx,
                        "ch": out_ch,   # out==in for depthwise
                    })

        elif lt == "Pointwise":
            for oc in layer_obj.get("out_channels", []):
                out_ch = int(oc.get("out_ch"))
                out_block = out_ch // 32  # <-- NEW: collapse 32 outputs

                for ic in oc.get("in_channels", []):
                    prev_in = int(ic.get("in_ch"))     # prev-layer input channel
                    w = prev_in // 32                  # <-- your requested divide-by-32 weight

                    for t in as_list(ic.get("starts")):
                        if t is None:
                            continue
                        t = int(t)
                        pw_group[(t, layer_idx, out_block, w)] += 1

    # Emit condensed pointwise entries
    for (t, layer_idx, out_block, w), count in pw_group.items():
        events.append({
            "time": t,
            "kind": "PW",
            "layer": layer_idx,
            "out_block": out_block,  # 0 = outs 0..31, 1 = outs 32..63, etc.
            "weight": w,             # prev_in_ch block (0 = 0..31, 1 = 32..63, etc.)
            "count": count,          # how many raw weights collapsed into this entry
        })

    # Sort by time (ascending). Tie-break: DW before PW, then fields
    def sort_key(e):
        kind_rank = 0 if e["kind"] == "DW" else 1
        return (
            e["time"],
            kind_rank,
            e.get("layer", 0),
            e.get("ch", -1),
            e.get("out_block", -1),
            e.get("weight", -1),
        )

    events.sort(key=sort_key)

    # Write JSON
    Path(args.out_json).write_text(json.dumps(events, indent=2), encoding="utf-8")

    # Write TXT
    lines: List[str] = []
    current_time = None
    printed_at_time = 0

    for e in events:
        t = e["time"]
        if t != current_time:
            current_time = t
            printed_at_time = 0
            lines.append(f"\n=== t = {t} ===")

        if args.max_per_time > 0 and printed_at_time >= args.max_per_time:
            continue

        if e["kind"] == "DW":
            lines.append(f"DW  L{e['layer']}  ch={e['ch']}")
        else:
            # out_block instead of out_ch
            lines.append(
                f"PW  L{e['layer']}  out_block={e['out_block']}  w={e['weight']}  (collapsed {e['count']})"
            )

        printed_at_time += 1

    Path(args.out_txt).write_text("\n".join(lines).lstrip(), encoding="utf-8")

    print(f"Wrote {args.out_json} ({len(events)} events)")
    print(f"Wrote {args.out_txt}")


if __name__ == "__main__":
    main()
