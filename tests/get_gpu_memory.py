from pynvml import smi as nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

ans = 0
while(True):
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print(mem_res.used / (1024**2)) # usage in GiB
    if (mem_res.used / (1024**2) > ans):
        ans = mem_res.used / (1024**2)
        print(ans)
# print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
