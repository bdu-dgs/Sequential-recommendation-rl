"""Build all processed tables from the raw KuaiRec CSVs.  Usage: python scripts/prepare_data.py"""

import json

from seqrec_rl.data.prepare import prepare

if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
