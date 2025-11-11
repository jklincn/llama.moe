from pathlib import Path
import subprocess


base_dir = Path(__file__).resolve().parent
repo_dir = base_dir.parent.parent.parent
build_path = repo_dir / "build"


def run_process(cmd):
    process = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def get_memory_bandwidth():
    """
    返回内存最大带宽，单位 MB/s
    """
    total = []

    def once():
        nonlocal total
        cmd = [
            "numactl",
            "--cpunodebind=0,1",
            "--membind=0,1",
            "env",
            "OMP_NUM_THREADS=48",
            "OMP_PLACES=cores",
            "OMP_PROC_BIND=spread",
            str(build_path / "stream" / "stream"),
        ]
        return_code, stdout, stderr = run_process(cmd)
        if return_code != 0:
            raise RuntimeError(f"Error running stream: {stderr}")
        result = next(
            line for line in stdout.splitlines() if line.strip().startswith("Copy:")
        )
        total.append(float(result.split()[1]))

    for _ in range(5):
        once()
    avg = sum(total) / len(total)
    return avg


def get_pcie_bandwidth():
    """
    返回 PCIe 最大带宽，单位 GB/s
    """
    total = []

    def once():
        nonlocal total
        cmd = [
            str(build_path / "nvbandwidth" / "nvbandwidth"),
            "-t",
            "host_to_device_memcpy_ce",
        ]
        return_code, stdout, stderr = run_process(cmd)
        if return_code != 0:
            raise RuntimeError(f"Error running nvbandwidth: {stderr}")
        result = next(
            line
            for line in stdout.splitlines()
            if line.strip().startswith("SUM host_to_device_memcpy_ce")
        )
        total.append(float(result.split()[-1]))

    for _ in range(5):
        once()
    avg = sum(total) / len(total)
    return avg


def main():
    mem_max_bw = get_memory_bandwidth()
    print(f"Memory Bandwidth: {mem_max_bw / 1e3:0.2f} GB/s")

    pcie_max_bw = get_pcie_bandwidth()
    print(f"PCIe Bandwidth: {pcie_max_bw:0.2f} GB/s")


if __name__ == "__main__":
    main()
