#!/usr/bin/env python3

from __future__ import annotations

import sys
import json
import traceback
from pathlib import Path
import time


from runtime import ONNXRunner
from runtime.io import (
    load_manifest,
    save_output,
)


class InferenceWorker:


    def __init__(self):

        # model cache
        self.runners = {}


    def log(self, msg):

        print(
            msg,
            file=sys.stderr,
            flush=True,
        )


    def initialize(self):

        self.log(
            "[worker] initialized"
        )

        print(
            "READY",
            flush=True,
        )


    def get_runner(
        self,
        model_path,
        batch_size,
    ):


        key = (
            str(model_path),
            batch_size,
        )


        if key not in self.runners:

            self.log(
                f"[worker] loading {model_path}"
            )


            self.runners[key] = ONNXRunner(
                model_path,
                batch_size=batch_size,
            )


        return self.runners[key]



    def run_task(
        self,
        task,
    ):

        model_path = task["onnx"]

        input_dir = task["input"]

        output_dir = task["output"]

        batch_size = task.get(
            "batch_size",
            256,
        )


        runner = self.get_runner(
            model_path,
            batch_size,
        )


        # -------------------------
        # load input
        # -------------------------

        manifest, tensors = load_manifest(
            input_dir
        )
        start=time.perf_counter()


        # -------------------------
        # inference
        # -------------------------

        outputs = runner.run(
            tensors
        )
        end=time.perf_counter()


        latency_ms = (
            end-start
        )*1000


        self.log(
            f"total inference latency: {latency_ms:.3f} ms"
        )



        # -------------------------
        # save output
        # -------------------------

        save_output(
            output_dir,
            outputs,
        )


        # samples
        first_tensor = next(
            iter(tensors.values())
        )


        return {
            "samples":
                int(first_tensor.shape[0]),

            "outputs":
                list(outputs.keys()),
        }



    def loop(self):

        self.initialize()


        for line in sys.stdin:


            line=line.strip()


            if not line:
                continue



            try:

                task=json.loads(
                    line
                )


                if task.get("cmd")=="exit":

                    self.log(
                        "[worker] exit"
                    )

                    break



                result=self.run_task(
                    task
                )


                response={
                    "status":"ok",
                    **result,
                }


                print(
                    json.dumps(response),
                    flush=True,
                )



            except Exception as e:


                traceback.print_exc(
                    file=sys.stderr
                )


                response={
                    "status":"error",
                    "error":str(e),
                }


                print(
                    json.dumps(response),
                    flush=True,
                )





if __name__=="__main__":

    worker=InferenceWorker()

    worker.loop()