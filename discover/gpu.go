package discover

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.
// Included to drive logic for reducing Ollama-allocated overhead on L4T/Jetson devices.
var CudaTegra string = os.Getenv("JETSON_JETPACK")

func GetCPUInfo() GpuInfo {
	mem, err := GetCPUMem()
	if err != nil {
		slog.Warn("error looking up system memory", "error", err)
	}

	return GpuInfo{
		memInfo: mem,
		DeviceID: ml.DeviceID{
			Library: "cpu",
			ID:      "0",
		},
	}
}

func GetGPUInfo(ctx context.Context, runners []FilteredRunnerDiscovery) GpuInfoList {
	devs := GPUDevices(ctx, runners)
	return devInfoToInfoList(devs)
}

const (
	cudaMinimumMemory = 457 * format.MebiByte
	rocmMinimumMemory = 457 * format.MebiByte
	// TODO OneAPI minimum memory
)

var (
	gpuMutex      sync.Mutex
	bootstrapped  bool
	cpus          []CPUInfo
	cudaGPUs      []CudaGPUInfo
	nvcudaLibPath string
	cudartLibPath string
	oneapiLibPath string
	nvmlLibPath   string
	rocmGPUs      []RocmGPUInfo
	oneapiGPUs    []OneapiGPUInfo

	// If any discovered GPUs are incompatible, report why
	unsupportedGPUs []UnsupportedGPUInfo

	// Keep track of errors during bootstrapping so that if GPUs are missing
	// they expected to be present this may explain why
	bootstrapErrors []error
)

// With our current CUDA compile flags, older than 5.0 will not work properly
// (string values used to allow ldflags overrides at build time)
var (
	CudaComputeMajorMin = "3"
	CudaComputeMinorMin = "5"
)

var RocmComputeMajorMin = "9"

// TODO find a better way to detect iGPU instead of minimum memory
const IGPUMemLimit = 1 * format.GibiByte // 512G is what they typically report, so anything less than 1G must be iGPU

// Note: gpuMutex must already be held
func initCudaHandles() *cudaHandles {
	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	cHandles := &cudaHandles{}
	// Short Circuit if we already know which library to use
	// ignore bootstrap errors in this case since we already recorded them
	if nvmlLibPath != "" {
		cHandles.nvml, _, _ = loadNVMLMgmt([]string{nvmlLibPath})
		return cHandles
	}
	if nvcudaLibPath != "" {
		cHandles.deviceCount, cHandles.nvcuda, _, _ = loadNVCUDAMgmt([]string{nvcudaLibPath})
		return cHandles
	}
	if cudartLibPath != "" {
		cHandles.deviceCount, cHandles.cudart, _, _ = loadCUDARTMgmt([]string{cudartLibPath})
		return cHandles
	}

	slog.Debug("searching for GPU discovery libraries for NVIDIA")
	var cudartMgmtPatterns []string

	// Aligned with driver, we can't carry as payloads
	nvcudaMgmtPatterns := NvcudaGlobs
	cudartMgmtPatterns = append(cudartMgmtPatterns, filepath.Join(LibOllamaPath, "cuda_v*", CudartMgmtName))
	cudartMgmtPatterns = append(cudartMgmtPatterns, CudartGlobs...)

	if len(NvmlGlobs) > 0 {
		nvmlLibPaths := FindGPULibs(NvmlMgmtName, NvmlGlobs)
		if len(nvmlLibPaths) > 0 {
			nvml, libPath, err := loadNVMLMgmt(nvmlLibPaths)
			if nvml != nil {
				slog.Debug("nvidia-ml loaded", "library", libPath)
				cHandles.nvml = nvml
				nvmlLibPath = libPath
			}
			if err != nil {
				bootstrapErrors = append(bootstrapErrors, err)
			}
		}
	}

	nvcudaLibPaths := FindGPULibs(NvcudaMgmtName, nvcudaMgmtPatterns)
	if len(nvcudaLibPaths) > 0 {
		deviceCount, nvcuda, libPath, err := loadNVCUDAMgmt(nvcudaLibPaths)
		if nvcuda != nil {
			slog.Debug("detected GPUs", "count", deviceCount, "library", libPath)
			cHandles.nvcuda = nvcuda
			cHandles.deviceCount = deviceCount
			nvcudaLibPath = libPath
			return cHandles
		}
		if err != nil {
			bootstrapErrors = append(bootstrapErrors, err)
		}
	}

	cudartLibPaths := FindGPULibs(CudartMgmtName, cudartMgmtPatterns)
	if len(cudartLibPaths) > 0 {
		deviceCount, cudart, libPath, err := loadCUDARTMgmt(cudartLibPaths)
		if cudart != nil {
			slog.Debug("detected GPUs", "library", libPath, "count", deviceCount)
			cHandles.cudart = cudart
			cHandles.deviceCount = deviceCount
			cudartLibPath = libPath
			return cHandles
		}
		if err != nil {
			bootstrapErrors = append(bootstrapErrors, err)
		}
	}

	return cHandles
}

// Note: gpuMutex must already be held
func initOneAPIHandles() *oneapiHandles {
	oHandles := &oneapiHandles{}

	// Short Circuit if we already know which library to use
	// ignore bootstrap errors in this case since we already recorded them
	if oneapiLibPath != "" {
		oHandles.deviceCount, oHandles.oneapi, _, _ = loadOneapiMgmt([]string{oneapiLibPath})
		return oHandles
	}

	oneapiLibPaths := FindGPULibs(OneapiMgmtName, OneapiGlobs)
	if len(oneapiLibPaths) > 0 {
		var err error
		oHandles.deviceCount, oHandles.oneapi, oneapiLibPath, err = loadOneapiMgmt(oneapiLibPaths)
		if err != nil {
			bootstrapErrors = append(bootstrapErrors, err)
		}
	}

	return oHandles
}

func GetCPUInfo() GpuInfoList {
	gpuMutex.Lock()
	if !bootstrapped {
		gpuMutex.Unlock()
		GetGPUInfo()
	} else {
		gpuMutex.Unlock()
	}
	return GpuInfoList{cpus[0].GpuInfo}
}

func GetGPUInfo() GpuInfoList {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	needRefresh := true
	var cHandles *cudaHandles
	var oHandles *oneapiHandles
	defer func() {
		if cHandles != nil {
			if cHandles.cudart != nil {
				C.cudart_release(*cHandles.cudart)
			}
			if cHandles.nvcuda != nil {
				C.nvcuda_release(*cHandles.nvcuda)
			}
			if cHandles.nvml != nil {
				C.nvml_release(*cHandles.nvml)
			}
		}
		if oHandles != nil {
			if oHandles.oneapi != nil {
				// TODO - is this needed?
				C.oneapi_release(*oHandles.oneapi)
			}
		}
	}()

	if !bootstrapped {
		slog.Info("looking for compatible GPUs")
		cudaComputeMajorMin, err := strconv.Atoi(CudaComputeMajorMin)
		if err != nil {
			slog.Error("invalid CudaComputeMajorMin setting", "value", CudaComputeMajorMin, "error", err)
		}
		cudaComputeMinorMin, err := strconv.Atoi(CudaComputeMinorMin)
		if err != nil {
			slog.Error("invalid CudaComputeMinorMin setting", "value", CudaComputeMinorMin, "error", err)
		}
		bootstrapErrors = []error{}
		needRefresh = false
		var memInfo C.mem_info_t

		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		}

		details, err := GetCPUDetails()
		if err != nil {
			slog.Warn("failed to lookup CPU details", "error", err)
		}
		cpus = []CPUInfo{
			{
				GpuInfo: GpuInfo{
					memInfo: mem,
					Library: "cpu",
					ID:      "0",
				},
				CPUs: details,
			},
		}

		// Load ALL libraries
		cHandles = initCudaHandles()

		// NVIDIA
		for i := range cHandles.deviceCount {
			if cHandles.cudart != nil || cHandles.nvcuda != nil {
				gpuInfo := CudaGPUInfo{
					GpuInfo: GpuInfo{
						Library: "cuda",
					},
					index: i,
				}
				var driverMajor int
				var driverMinor int
				if cHandles.cudart != nil {
					C.cudart_bootstrap(*cHandles.cudart, C.int(i), &memInfo)
				} else {
					C.nvcuda_bootstrap(*cHandles.nvcuda, C.int(i), &memInfo)
					driverMajor = int(cHandles.nvcuda.driver_major)
					driverMinor = int(cHandles.nvcuda.driver_minor)
				}
				if memInfo.err != nil {
					slog.Info("error looking up nvidia GPU memory", "error", C.GoString(memInfo.err))
					C.free(unsafe.Pointer(memInfo.err))
					continue
				}
				gpuInfo.TotalMemory = uint64(memInfo.total)
				gpuInfo.FreeMemory = uint64(memInfo.free)
				gpuInfo.ID = C.GoString(&memInfo.gpu_id[0])
				gpuInfo.Compute = fmt.Sprintf("%d.%d", memInfo.major, memInfo.minor)
				gpuInfo.computeMajor = int(memInfo.major)
				gpuInfo.computeMinor = int(memInfo.minor)
				gpuInfo.MinimumMemory = cudaMinimumMemory
				gpuInfo.DriverMajor = driverMajor
				gpuInfo.DriverMinor = driverMinor
				variant := cudaVariant(gpuInfo)

				// Start with our bundled libraries
				if variant != "" {
					variantPath := filepath.Join(LibOllamaPath, "cuda_"+variant)
					if _, err := os.Stat(variantPath); err == nil {
						// Put the variant directory first in the search path to avoid runtime linking to the wrong library
						gpuInfo.DependencyPath = append([]string{variantPath}, gpuInfo.DependencyPath...)
					}
				}
				gpuInfo.Name = C.GoString(&memInfo.gpu_name[0])
				gpuInfo.Variant = variant

				if int(memInfo.major) < cudaComputeMajorMin || (int(memInfo.major) == cudaComputeMajorMin && int(memInfo.minor) < cudaComputeMinorMin) {
					unsupportedGPUs = append(unsupportedGPUs,
						UnsupportedGPUInfo{
							GpuInfo: gpuInfo.GpuInfo,
						})
					slog.Info(fmt.Sprintf("[%d] CUDA GPU is too old. Compute Capability detected: %d.%d", i, memInfo.major, memInfo.minor))
					continue
				}

				// query the management library as well so we can record any skew between the two
				// which represents overhead on the GPU we must set aside on subsequent updates
				if cHandles.nvml != nil {
					uuid := C.CString(gpuInfo.ID)
					defer C.free(unsafe.Pointer(uuid))
					C.nvml_get_free(*cHandles.nvml, uuid, &memInfo.free, &memInfo.total, &memInfo.used)
					if memInfo.err != nil {
						slog.Warn("error looking up nvidia GPU memory", "error", C.GoString(memInfo.err))
						C.free(unsafe.Pointer(memInfo.err))
					} else {
						if memInfo.free != 0 && uint64(memInfo.free) > gpuInfo.FreeMemory {
							gpuInfo.OSOverhead = uint64(memInfo.free) - gpuInfo.FreeMemory
							slog.Info("detected OS VRAM overhead",
								"id", gpuInfo.ID,
								"library", gpuInfo.Library,
								"compute", gpuInfo.Compute,
								"driver", fmt.Sprintf("%d.%d", gpuInfo.DriverMajor, gpuInfo.DriverMinor),
								"name", gpuInfo.Name,
								"overhead", format.HumanBytes2(gpuInfo.OSOverhead),
							)
						}
					}
				}

				// TODO potentially sort on our own algorithm instead of what the underlying GPU library does...
				cudaGPUs = append(cudaGPUs, gpuInfo)
			}
		}

		// Intel
		if envconfig.IntelGPU() {
			oHandles = initOneAPIHandles()
			if oHandles != nil && oHandles.oneapi != nil {
				for d := range oHandles.oneapi.num_drivers {
					if oHandles.oneapi == nil {
						// shouldn't happen
						slog.Warn("nil oneapi handle with driver count", "count", int(oHandles.oneapi.num_drivers))
						continue
					}
					devCount := C.oneapi_get_device_count(*oHandles.oneapi, C.int(d))
					for i := range devCount {
						gpuInfo := OneapiGPUInfo{
							GpuInfo: GpuInfo{
								Library: "oneapi",
							},
							driverIndex: int(d),
							gpuIndex:    int(i),
						}
						// TODO - split bootstrapping from updating free memory
						C.oneapi_check_vram(*oHandles.oneapi, C.int(d), i, &memInfo)
						// TODO - convert this to MinimumMemory based on testing...
						var totalFreeMem float64 = float64(memInfo.free) * 0.95 // work-around: leave some reserve vram for mkl lib used in ggml-sycl backend.
						memInfo.free = C.uint64_t(totalFreeMem)
						gpuInfo.TotalMemory = uint64(memInfo.total)
						gpuInfo.FreeMemory = uint64(memInfo.free)
						gpuInfo.ID = C.GoString(&memInfo.gpu_id[0])
						gpuInfo.Name = C.GoString(&memInfo.gpu_name[0])
						gpuInfo.DependencyPath = []string{LibOllamaPath}
						oneapiGPUs = append(oneapiGPUs, gpuInfo)
					}
				}
			}
		}

		rocmGPUs, err = AMDGetGPUInfo()
		if err != nil {
			bootstrapErrors = append(bootstrapErrors, err)
		}
		bootstrapped = true
		if len(cudaGPUs) == 0 && len(rocmGPUs) == 0 && len(oneapiGPUs) == 0 {
			slog.Info("no compatible GPUs were discovered")
		}

		// TODO verify we have runners for the discovered GPUs, filter out any that aren't supported with good error messages
	}

	// For detected GPUs, load library if not loaded

	// Refresh free memory usage
	if needRefresh {
		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		} else {
			slog.Debug("updating system memory data",
				slog.Group(
					"before",
					"total", format.HumanBytes2(cpus[0].TotalMemory),
					"free", format.HumanBytes2(cpus[0].FreeMemory),
					"free_swap", format.HumanBytes2(cpus[0].FreeSwap),
				),
				slog.Group(
					"now",
					"total", format.HumanBytes2(mem.TotalMemory),
					"free", format.HumanBytes2(mem.FreeMemory),
					"free_swap", format.HumanBytes2(mem.FreeSwap),
				),
			)
			cpus[0].FreeMemory = mem.FreeMemory
			cpus[0].FreeSwap = mem.FreeSwap
		}

		var memInfo C.mem_info_t
		if cHandles == nil && len(cudaGPUs) > 0 {
			cHandles = initCudaHandles()
		}
		for i, gpu := range cudaGPUs {
			if cHandles.nvml != nil {
				uuid := C.CString(gpu.ID)
				defer C.free(unsafe.Pointer(uuid))
				C.nvml_get_free(*cHandles.nvml, uuid, &memInfo.free, &memInfo.total, &memInfo.used)
			} else if cHandles.cudart != nil {
				C.cudart_bootstrap(*cHandles.cudart, C.int(gpu.index), &memInfo)
			} else if cHandles.nvcuda != nil {
				C.nvcuda_get_free(*cHandles.nvcuda, C.int(gpu.index), &memInfo.free, &memInfo.total)
				memInfo.used = memInfo.total - memInfo.free
			} else {
				// shouldn't happen
				slog.Warn("no valid cuda library loaded to refresh vram usage")
				break
			}
			if memInfo.err != nil {
				slog.Warn("error looking up nvidia GPU memory", "error", C.GoString(memInfo.err))
				C.free(unsafe.Pointer(memInfo.err))
				continue
			}
			if memInfo.free == 0 {
				slog.Warn("error looking up nvidia GPU memory")
				continue
			}
			if cHandles.nvml != nil && gpu.OSOverhead > 0 {
				// When using the management library update based on recorded overhead
				memInfo.free -= C.uint64_t(gpu.OSOverhead)
			}
			slog.Debug("updating cuda memory data",
				"gpu", gpu.ID,
				"name", gpu.Name,
				"overhead", format.HumanBytes2(gpu.OSOverhead),
				slog.Group(
					"before",
					"total", format.HumanBytes2(gpu.TotalMemory),
					"free", format.HumanBytes2(gpu.FreeMemory),
				),
				slog.Group(
					"now",
					"total", format.HumanBytes2(uint64(memInfo.total)),
					"free", format.HumanBytes2(uint64(memInfo.free)),
					"used", format.HumanBytes2(uint64(memInfo.used)),
				),
			)
			cudaGPUs[i].FreeMemory = uint64(memInfo.free)
		}

		if oHandles == nil && len(oneapiGPUs) > 0 {
			oHandles = initOneAPIHandles()
		}
		for i, gpu := range oneapiGPUs {
			if oHandles.oneapi == nil {
				// shouldn't happen
				slog.Warn("nil oneapi handle with device count", "count", oHandles.deviceCount)
				continue
			}
			C.oneapi_check_vram(*oHandles.oneapi, C.int(gpu.driverIndex), C.int(gpu.gpuIndex), &memInfo)
			// TODO - convert this to MinimumMemory based on testing...
			var totalFreeMem float64 = float64(memInfo.free) * 0.95 // work-around: leave some reserve vram for mkl lib used in ggml-sycl backend.
			memInfo.free = C.uint64_t(totalFreeMem)
			oneapiGPUs[i].FreeMemory = uint64(memInfo.free)
		}

		err = RocmGPUInfoList(rocmGPUs).RefreshFreeMemory()
		if err != nil {
			slog.Debug("problem refreshing ROCm free memory", "error", err)
		}
	}


func devInfoToInfoList(devs []ml.DeviceInfo) GpuInfoList {
	resp := []GpuInfo{}
	// Our current packaging model places ggml-hip in the main directory
	// but keeps rocm in an isolated directory.  We have to add it to
	// the [LD_LIBRARY_]PATH so ggml-hip will load properly
	rocmDir := filepath.Join(LibOllamaPath, "rocm")
	if _, err := os.Stat(rocmDir); err != nil {
		rocmDir = ""
	}

	for _, dev := range devs {
		info := GpuInfo{
			DeviceID: dev.DeviceID,
			filterID: dev.FilteredID,
			Name:     dev.Description,
			memInfo: memInfo{
				TotalMemory: dev.TotalMemory,
				FreeMemory:  dev.FreeMemory,
			},
			// TODO can we avoid variant
			DependencyPath: dev.LibraryPath,
			DriverMajor:    dev.DriverMajor,
			DriverMinor:    dev.DriverMinor,
			ComputeMajor:   dev.ComputeMajor,
			ComputeMinor:   dev.ComputeMinor,
		}
		if dev.Library == "CUDA" || dev.Library == "ROCm" {
			info.MinimumMemory = 457 * format.MebiByte
		}
		if dev.Library == "ROCm" && rocmDir != "" {
			info.DependencyPath = append(info.DependencyPath, rocmDir)
		}
		resp = append(resp, info)
	}
	if len(resp) == 0 {
		mem, err := GetCPUMem()
		if err != nil {
			slog.Warn("error looking up system memory", "error", err)
		}

		resp = append(resp, GpuInfo{
			memInfo: mem,
			DeviceID: ml.DeviceID{
				Library: "cpu",
				ID:      "0",
			},
		})
	}
	return resp
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variable
//
// If different libraries are detected, the first one is what we use
func (l GpuInfoList) GetVisibleDevicesEnv() []string {
	if len(l) == 0 {
		return nil
	}
	return []string{rocmGetVisibleDevicesEnv(l)}
}

func rocmGetVisibleDevicesEnv(gpuInfo []GpuInfo) string {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "ROCm" {
			continue
		}
		// If the devices requires a numeric ID, for filtering purposes, we use the unfiltered ID number
		if info.filterID != "" {
			ids = append(ids, info.filterID)
		} else {
			ids = append(ids, info.ID)
		}
	}
	if len(ids) == 0 {
		return ""
	}
	envVar := "ROCR_VISIBLE_DEVICES="
	if runtime.GOOS != "linux" {
		envVar = "HIP_VISIBLE_DEVICES="
	}
	// There are 3 potential env vars to use to select GPUs.
	// ROCR_VISIBLE_DEVICES supports UUID or numeric but does not work on Windows
	// HIP_VISIBLE_DEVICES supports numeric IDs only
	// GPU_DEVICE_ORDINAL supports numeric IDs only
	return envVar + strings.Join(ids, ",")
}

// GetSystemInfo returns the last cached state of the GPUs on the system
func GetSystemInfo() SystemInfo {
	deviceMu.Lock()
	defer deviceMu.Unlock()
	gpus := devInfoToInfoList(devices)
	if len(gpus) == 1 && gpus[0].Library == "cpu" {
		gpus = []GpuInfo{}
	}

	return SystemInfo{
		System: CPUInfo{
			CPUs:    GetCPUDetails(),
			GpuInfo: GetCPUInfo(),
		},
		GPUs: gpus,
	}
}

func cudaJetpack() string {
	if runtime.GOARCH == "arm64" && runtime.GOOS == "linux" {
		if CudaTegra != "" {
			ver := strings.Split(CudaTegra, ".")
			if len(ver) > 0 {
				return "jetpack" + ver[0]
			}
		} else if data, err := os.ReadFile("/etc/nv_tegra_release"); err == nil {
			r := regexp.MustCompile(` R(\d+) `)
			m := r.FindSubmatch(data)
			if len(m) != 2 {
				slog.Info("Unexpected format for /etc/nv_tegra_release.  Set JETSON_JETPACK to select version")
			} else {
				if l4t, err := strconv.Atoi(string(m[1])); err == nil {
					// Note: mapping from L4t -> JP is inconsistent (can't just subtract 30)
					// https://developer.nvidia.com/embedded/jetpack-archive
					switch l4t {
					case 35:
						return "jetpack5"
					case 36:
						return "jetpack6"
					default:
						// Newer Jetson systems use the SBSU runtime
						slog.Debug("unrecognized L4T version", "nv_tegra_release", string(data))
					}
				}
			}
		}
	}
	return ""
}
