import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Minimal pass-through target model for local validation demos."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_cfg = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_cfg["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            batch = input0.as_numpy().shape[0]
            output = np.zeros((batch, 1000), dtype=self.output_dtype)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT0", output)]
                )
            )
        return responses
