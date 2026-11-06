"""block_sensitivity.py sweep for RT-DETR-l through the adapters of rd_common (rules_11_12d.md)."""
import sys

import rd_common  # noqa: F401  (must come first)
import block_sensitivity as bs

assert bs.blocks is rd_common.blocks_rd and bs.run is rd_common.run_rd
sys.argv = [sys.argv[0], "--model", "rtdetr-l", "--base", "int8ent"]
bs.main()
