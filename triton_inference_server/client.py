from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_classifier(input: str) -> dict:
    triton_client = get_client()

    model_input = InferInput(name="float_input", shape=input.shape, datatype="FP32")
    model_input.set_data_from_numpy(input, binary_data=True)

    query_response = triton_client.infer(
        "onnx_classifier",
        [model_input],
        outputs=[
            InferRequestedOutput("probabilities", binary_data=True),
            InferRequestedOutput("label", binary_data=True)
        ],
    )

    dict_response = {
        "probabilities": query_response.as_numpy("probabilities"),
        "label": query_response.as_numpy("label")
    }
    return dict_response


def main():
    input = np.array(
        [
            [6.1, 2.8, 4.0, 1.3],
            [4.8, 3.4, 1.9, 0.2],
            [5.0, 2.0, 3.5, 1.0],
            [4.3, 3.0, 1.1, 0.1],
            [5.4, 3.4, 1.5, 0.4],
            [5.1, 3.3, 1.7, 0.5],
            [6.0, 3.0, 4.8, 1.8],
        ],
        dtype=np.float32,
    )

    output = call_triton_classifier(input)
    print(f"Probabilities:\n{output['probabilities']}")
    print(f"Labels:\n{output['label']}")


if __name__ == "__main__":
    main()
