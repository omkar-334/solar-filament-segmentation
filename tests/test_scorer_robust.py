"""scorer_robust: every instance is one 8-connected piece and no two instances touch."""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_filament.postproc import EIGHT, scorer_robust  # noqa: E402


def test_split_and_touching_instances():
    lab = np.zeros((200, 200), np.uint16)
    lab[10:60, 10:40] = 1          # big piece of instance 1
    lab[100:110, 100:110] = 1      # small detached piece of instance 1
    lab[10:60, 40:80] = 2          # instance 2 touches instance 1
    out = scorer_robust(lab, min_area=20)
    assert out.max() == 2
    for i in (1, 2):
        assert ndimage.label(out == i, structure=EIGHT)[1] == 1
    assert out[100:110, 100:110].max() == 0
    # Union connected components must equal the instance count (no touching).
    assert ndimage.label(out > 0, structure=EIGHT)[1] == 2
