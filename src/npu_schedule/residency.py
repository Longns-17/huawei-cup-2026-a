"""Exact serial-visit residency proxy using explicit CPU or CUDA execution.

CUDA uses three stages: tensor intervals -> parallel tile scans -> tile offsets.
No automatic device selection or fallback is performed.
"""

import os
import time

import numpy as np

from .problem import ROOT

CUDA_SOURCE = r"""
extern "C" __global__ void tensor_events(
 const int* source,const int* indptr,const int* readers,const int* memory,
 const long long* bytes,const int* location,int nt,int nv,int nk,int width,
 unsigned long long* changes) {
 int tensor=blockIdx.x*blockDim.x+threadIdx.x;
 int core=blockIdx.y;
 if(tensor>=nt) return;
 const int* where=location+core*nv;
 int begin=width, end=-1;
 if(source[tensor]>=0) {
   int p=where[source[tensor]];
   if(p>=0) { begin=p; end=p; }
 }
 for(int j=indptr[tensor]; j<indptr[tensor+1]; ++j) {
   int p=where[readers[j]];
   if(p>=0) {begin=p<begin?p:begin;end=p>end?p:end;}
 }
 if(end<0) return;
 int offset=(core*3+memory[tensor])*width;
 atomicAdd(changes+offset+begin,(unsigned long long)bytes[tensor]);
 atomicAdd(changes+offset+end+1,(unsigned long long)(-bytes[tensor]));
}
extern "C" __global__ void tile_prefix(
 const long long* changes,int width,int tiles,long long* totals,long long* maxima) {
 __shared__ long long work[256];
 int row=blockIdx.y, tile=blockIdx.x, thread=threadIdx.x;
 int column=tile*256+thread;
 work[thread]=column<width?changes[row*width+column]:0;
 __syncthreads();
 for(int distance=1;distance<256;distance<<=1) {
   long long value=thread>=distance?work[thread-distance]:0;
   __syncthreads(); work[thread]+=value; __syncthreads();
 }
 if(thread==255) totals[row*tiles+tile]=work[thread];
 __syncthreads();
 for(int stride=128;stride>0;stride>>=1) {
   if(thread<stride && work[thread+stride]>work[thread]) work[thread]=work[thread+stride];
   __syncthreads();
 }
 if(thread==0) maxima[row*tiles+tile]=work[0];
}
extern "C" __global__ void join_tiles(
 const long long* totals,const long long* maxima,int rows,int tiles,long long* peak) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows) return;
 long long offset=0, maximum=0;
 for(int tile=0;tile<tiles;++tile) {
   long long level=offset+maxima[row*tiles+tile];
   maximum=level>maximum?level:maximum;
   offset+=totals[row*tiles+tile];
 }
 peak[row]=maximum;
}
"""


class Residency:
    def __init__(self, problem, device):
        self.problem = problem
        self.device = device
        self.calls = 0
        self.seconds = 0.0
        self.upload_seconds = 0.0
        self.regions = ("L1", "UB", "DDR")
        self.indices = {region: i for i, region in enumerate(self.regions)}
        self.sources = np.array(
            [-1 if t.producer is None else t.producer for t in problem.tensors], dtype=np.int32
        )
        self.spaces = np.array([self.indices[t.region] for t in problem.tensors], dtype=np.int32)
        self.sizes = np.array([t.size for t in problem.tensors], dtype=np.int64)
        counts = np.array([len(t.consumers) for t in problem.tensors], dtype=np.int32)
        self.indptr = np.r_[np.int32(0), np.cumsum(counts, dtype=np.int32)]
        self.readers = np.array([v for t in problem.tensors for v in sorted(t.consumers)], dtype=np.int32)
        if device == "cuda":
            started = time.perf_counter()
            os.environ["CUPY_CACHE_DIR"] = str(ROOT / "reports/cuda_kernel_cache")
            import cupy

            self.cp = cupy
            self.cp.cuda.Device(0).use()
            module = cupy.RawModule(code=CUDA_SOURCE, options=("--std=c++11",))
            self.events_kernel = module.get_function("tensor_events")
            self.prefix_kernel = module.get_function("tile_prefix")
            self.join_kernel = module.get_function("join_tiles")
            self.gpu_inputs = tuple(
                cupy.asarray(a) for a in (self.sources, self.indptr, self.readers, self.spaces, self.sizes)
            )
            self.cp.cuda.get_current_stream().synchronize()
            self.upload_seconds = time.perf_counter() - started
            self.device_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        elif device == "cpu":
            self.device_name = "CPU"
        else:
            raise ValueError("Choose cpu or cuda explicitly")

    def compute(self, sequences):
        started = time.perf_counter()
        n = len(self.problem.identifiers)
        k = len(sequences)
        positions = np.full((k, n), -1, dtype=np.int32)
        for c, seq in enumerate(sequences):
            positions[c, list(seq)] = np.arange(len(seq), dtype=np.int32)
        width = max(map(len, sequences)) + 1
        if self.device == "cpu":
            delta = np.zeros((k, 3, width), dtype=np.int64)
            for c in range(k):
                for t, tensor in enumerate(self.problem.tensors):
                    visits = [positions[c, v] for v in tensor.consumers if positions[c, v] >= 0]
                    if tensor.producer is not None and positions[c, tensor.producer] >= 0:
                        visits.append(positions[c, tensor.producer])
                    if visits:
                        delta[c, self.spaces[t], min(visits)] += tensor.size
                        delta[c, self.spaces[t], max(visits) + 1] -= tensor.size
            peaks = np.cumsum(delta, axis=2).max(axis=2)
        else:
            cp = self.cp
            delta = cp.zeros((k, 3, width), dtype=cp.int64)
            tiles = (width + 255) // 256
            totals = cp.empty((k * 3, tiles), dtype=cp.int64)
            maxima = cp.empty((k * 3, tiles), dtype=cp.int64)
            peak = cp.empty((k, 3), dtype=cp.int64)
            tensors = len(self.problem.tensors)
            self.events_kernel(
                ((tensors + 255) // 256, k),
                (256,),
                (
                    *self.gpu_inputs,
                    cp.asarray(positions),
                    np.int32(tensors),
                    np.int32(n),
                    np.int32(k),
                    np.int32(width),
                    delta,
                ),
            )
            self.prefix_kernel(
                (tiles, k * 3), (256,), (delta, np.int32(width), np.int32(tiles), totals, maxima)
            )
            self.join_kernel((1,), (32,), (totals, maxima, np.int32(k * 3), np.int32(tiles), peak))
            peaks = peak.get()
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return [{region: int(peaks[c, j]) for j, region in enumerate(self.regions)} for c in range(k)]

    def report(self):
        return dict(
            device=self.device,
            device_name=self.device_name,
            scan_batches=self.calls,
            scan_seconds=self.seconds,
            initialization_seconds=self.upload_seconds,
        )
