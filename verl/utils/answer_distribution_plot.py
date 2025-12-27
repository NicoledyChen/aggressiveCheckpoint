# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Mapping, Sequence


def _sort_sample_id(sample_id: str):
    try:
        return int(sample_id)
    except Exception:
        return sample_id


def save_answer_distribution_grid_png(
    records_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    out_path: str,
    *,
    top_k: int = 8,
    max_questions: int = 50,
    title: str | None = None,
    columns: int | None = None,
    cell_size: tuple[int, int] = (260, 220),
) -> None:
    """Save a grid of per-question answer distributions as a PNG.

    `records_by_sample` should map sample_id -> list of records. Each record is expected to contain:
    - normalized_answer: str
    - ground_truth: str (optional, used for title)
    """
    # Lazy import so training can run without PIL if the user disables this feature.
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    if not records_by_sample:
        return

    sample_ids = sorted(records_by_sample.keys(), key=_sort_sample_id)
    if max_questions is not None and max_questions > 0:
        sample_ids = sample_ids[:max_questions]

    if not sample_ids:
        return

    n_questions = len(sample_ids)
    if columns is None or columns <= 0:
        columns = min(6, max(1, int(math.ceil(math.sqrt(n_questions)))))
    rows = int(math.ceil(n_questions / columns))

    cell_w, cell_h = cell_size
    header_h = 40 if title else 0
    img_w = columns * cell_w
    img_h = header_h + rows * cell_h

    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    if title:
        draw.text((10, 10), title, fill="black", font=font)

    for idx, sid in enumerate(sample_ids):
        recs = list(records_by_sample.get(sid, []))
        if not recs:
            continue

        row = idx // columns
        col = idx % columns
        x0 = col * cell_w
        y0 = header_h + row * cell_h

        # Cell border
        draw.rectangle([x0 + 1, y0 + 1, x0 + cell_w - 2, y0 + cell_h - 2], outline=(220, 220, 220), width=1)

        gt = str(recs[0].get("ground_truth", ""))
        # Prefer showing 1-indexed question number when sample_id is numeric
        try:
            qnum = int(sid) + 1
            cell_title = f"Q{qnum} gt={gt}"
        except Exception:
            cell_title = f"id={sid} gt={gt}"

        draw.text((x0 + 8, y0 + 6), cell_title[:40], fill="black", font=font)

        answers = [str(r.get("normalized_answer", "")) for r in recs]
        counter = Counter(answers)
        total = sum(counter.values())
        common = counter.most_common(max(1, top_k))

        labels_counts: list[tuple[str, int]] = list(common)
        if len(counter) > len(common):
            other = total - sum(c for _, c in common)
            if other > 0:
                labels_counts.append(("OTHER", other))

        # Plot area
        margin_x = 10
        bar_top = y0 + 28
        bar_bottom = y0 + cell_h - 38
        bar_left = x0 + margin_x
        bar_right = x0 + cell_w - margin_x

        y_max = max([c for _, c in labels_counts] + [1])
        num_bars = len(labels_counts)
        if num_bars <= 0:
            continue

        gap = 4
        avail_w = bar_right - bar_left
        bar_w = max(8, int((avail_w - gap * (num_bars - 1)) / num_bars))
        actual_total_w = bar_w * num_bars + gap * (num_bars - 1)
        start_x = bar_left + max(0, (avail_w - actual_total_w) // 2)
        avail_h = max(1, bar_bottom - bar_top)

        for j, (lab, cnt) in enumerate(labels_counts):
            bx0 = start_x + j * (bar_w + gap)
            bx1 = bx0 + bar_w
            h = int((cnt / y_max) * avail_h)
            by0 = bar_bottom - h

            # Color invalid answers red-ish.
            color = (70, 130, 180)  # steelblue
            if lab in ("", "[INVALID]", "INVALID"):
                color = (220, 20, 60)  # crimson

            draw.rectangle([bx0, by0, bx1, bar_bottom], fill=color)
            draw.text((bx0, max(bar_top, by0 - 12)), str(cnt), fill="black", font=font)

            lab_s = str(lab)
            if len(lab_s) > 10:
                lab_s = lab_s[:9] + "..."
            draw.text((bx0, bar_bottom + 2), lab_s, fill="black", font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


