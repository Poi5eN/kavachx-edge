"""Automated Qualcomm AI Hub benchmark runner for QCS6490 (Dragonwing RB3 Gen 2).

Compiles and profiles YOLO models directly on Qualcomm cloud hardware.
"""
import argparse
import csv
import os
import time
import qai_hub as hub


def monitor_job(job, desc="Job"):
    print(f"Waiting for {desc} ({job.job_id})...")
    status = job.wait(timeout=600)
    print(f"{desc} completed with status: {status}")
    return job


def run_pipeline(model_path, model_name, device_name="Dragonwing RB3 Gen 2 Vision Kit"):
    device = hub.Device(device_name)
    print(f"\n==========================================")
    print(f"Targeting Device: {device.name} (Chipset: QCS6490)")
    print(f"Model: {model_path} ({model_name})")
    print(f"==========================================")

    # 1. Compile Job
    print("\n1. Submitting Compile Job for QCS6490...")
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=device,
        name=f"compile_{model_name}_qcs6490",
        input_specs=dict(images=(1, 3, 640, 640)),
        options="--target_runtime qnn_lib_aarch64_linux"
    )
    print(f"Compile Job URL: {compile_job.url}")
    monitor_job(compile_job, desc="Compile Job")

    target_model = compile_job.get_target_model()

    # 2. Profile Job on Device
    print("\n2. Submitting Profile Job to QCS6490 Hardware...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name=f"profile_{model_name}_qcs6490"
    )
    print(f"Profile Job URL: {profile_job.url}")
    monitor_job(profile_job, desc="Profile Job")

    # 3. Extract Profile Results
    profile_data = profile_job.download_profile()
    execution_summary = profile_data.get("execution_summary", {})
    inference_time = execution_summary.get("inference_time", {})
    
    mean_ms = round(inference_time.get("average", 0) / 1000.0, 2)
    min_ms = round(inference_time.get("min", 0) / 1000.0, 2)
    max_ms = round(inference_time.get("max", 0) / 1000.0, 2)
    fps = round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0
    peak_mem_mb = round(execution_summary.get("peak_memory_bytes", 0) / (1024 * 1024), 1)

    print("\n--- RESULTS FROM QCS6490 HARDWARE ---")
    print(f"Mean Latency: {mean_ms} ms (Min: {min_ms} ms, Max: {max_ms} ms)")
    print(f"Throughput:   {fps} FPS")
    print(f"Peak Memory:  {peak_mem_mb} MB")

    row = {
        "label": f"{model_name}_qcs6490_npu",
        "mode": "yolo",
        "model": model_path,
        "imgsz": 640,
        "frames": execution_summary.get("estimated_inference_count", 100),
        "mean_ms": mean_ms,
        "p50_ms": mean_ms,
        "p95_ms": max_ms,
        "fps": fps,
        "proc_cpu_pct": 0.0,
        "sys_cpu_pct_avg": 0.0,
        "rss_mb": peak_mem_mb,
        "device": "QCS6490 Hexagon NPU",
        "note": "AI Hub QCS6490"
    }

    results_path = "results/benchmarks.csv"
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"Saved result to {results_path}")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run_pipeline(args.model, args.name)
