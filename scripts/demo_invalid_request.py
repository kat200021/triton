#!/usr/bin/env python3
"""Send a deliberately invalid tensor to shape_validator and print the error."""

import sys

import numpy as np
import tritonclient.http as httpclient


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "localhost:8000"
    client = httpclient.InferenceServerClient(url)

    # Valid rank for Triton, invalid channels vs example_model config (expects 3).
    bad = np.random.rand(1, 1, 224, 224).astype(np.float32)
    inputs = [httpclient.InferInput("INPUT0", bad.shape, "FP32")]
    inputs[0].set_data_from_numpy(bad)

    try:
        client.infer("shape_validator", inputs)
    except httpclient.InferenceServerException as exc:
        print(exc.message())
        return 1

    print("Request unexpectedly succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
