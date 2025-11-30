import { useEffect, useState, useMemo } from 'react';
import { ModelDetail, ModelConfiguration, DiffusionModelDetail, AnyModelConfiguration, ImageModelConfiguration, VideoModelConfiguration } from '../types';
import { TimeEnergyTradeoffChart, ITLPercentile } from './TimeEnergyTradeoffChart';
import { EnergyPerResponseChart } from './EnergyPerResponseChart';
import { isDiffusionTask } from '../config/tasks';

interface ModelDetailModalProps {
  modelId: string;
  task: string;
  isOpen: boolean;
  onClose: () => void;
  onAddToComparison?: (modelId: string) => void;
  currentConfig?: {
    gpu_model: string;
    num_gpus: number;
    max_num_seqs?: number | null;
    batch_size?: number;
  };
}

type AnyModelDetail = ModelDetail | DiffusionModelDetail;

type SortDirection = 'asc' | 'desc' | null;

export function ModelDetailModal({
  modelId,
  task,
  isOpen,
  onClose,
  onAddToComparison,
  currentConfig: _currentConfig,
}: ModelDetailModalProps) {
  const isDiffusion = isDiffusionTask(task);
  const [modelDetail, setModelDetail] = useState<AnyModelDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const defaultSortKey = task === 'text-to-image' ? 'energy_per_image_joules' : task === 'text-to-video' ? 'energy_per_video_joules' : 'energy_per_token_joules';
  const [sortKey, setSortKey] = useState<string | null>(defaultSortKey);
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [selectedNumGPUs, setSelectedNumGPUs] = useState<Set<number>>(new Set());
  const [displayedConfig, setDisplayedConfig] = useState<AnyModelConfiguration | null>(null);
  const [selectedPercentile, setSelectedPercentile] = useState<ITLPercentile>('p50');
  const [maxEnergyPerResponse, setMaxEnergyPerResponse] = useState<number>(0);

  // Helper to get energy value from any configuration type
  const getEnergyValue = (config: AnyModelConfiguration): number | undefined => {
    if ('energy_per_image_joules' in config) return config.energy_per_image_joules;
    if ('energy_per_video_joules' in config) return config.energy_per_video_joules;
    if ('energy_per_token_joules' in config) return (config as ModelConfiguration).energy_per_token_joules;
    return undefined;
  };

  // Close on ESC key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen || !modelId || !task) return;

    const loadModelDetail = async () => {
      setLoading(true);
      setError(null);
      try {
        // Derive detail file path from modelId and task
        const detailFilePath = `models/${modelId.replace('/', '__')}__${task}.json`;
        const response = await fetch(`${import.meta.env.BASE_URL}data/${detailFilePath}`);
        if (!response.ok) {
          throw new Error(`Failed to load model details: ${response.statusText}`);
        }
        const data = await response.json();
        setModelDetail(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load model details');
      } finally {
        setLoading(false);
      }
    };

    loadModelDetail();
  }, [isOpen, modelId, task]);

  // Initialize filters and max energy when model detail loads
  useEffect(() => {
    if (modelDetail) {
      const gpuModels = new Set(modelDetail.configurations.map(c => c.gpu_model));
      const numGPUs = new Set(modelDetail.configurations.map(c => c.num_gpus));
      setSelectedGPUs(gpuModels);
      setSelectedNumGPUs(numGPUs);

      // Initialize displayed config to highest energy
      if (modelDetail.configurations.length > 0) {
        const highestEnergyConfig = modelDetail.configurations.reduce((max, c) =>
          (getEnergyValue(c) ?? 0) > (getEnergyValue(max) ?? 0) ? c : max
        );
        setDisplayedConfig(highestEnergyConfig);
      }

      // Calculate max energy per response based on actual data across ALL configs
      // This is only calculated once when modelDetail loads to ensure X-axis stays constant
      if (!isDiffusionTask(task)) {
        let maxEnergy = 0;
        for (const config of modelDetail.configurations) {
          const llmConfig = config as ModelConfiguration;
          if (!llmConfig.output_length_distribution || !llmConfig.energy_per_token_joules) continue;
          const { bins, counts } = llmConfig.output_length_distribution;
          // Find last bin with non-zero count for this config
          for (let i = counts.length - 1; i >= 0; i--) {
            if (counts[i] > 0) {
              const configMaxEnergy = bins[i + 1] * llmConfig.energy_per_token_joules;
              maxEnergy = Math.max(maxEnergy, configMaxEnergy);
              break;
            }
          }
        }
        setMaxEnergyPerResponse(maxEnergy);
      }
    }
  }, [modelDetail, task]);

  // Extract available GPU models and num_gpus
  const availableGPUs = modelDetail
    ? Array.from(new Set(modelDetail.configurations.map(c => c.gpu_model))).sort()
    : [];
  const availableNumGPUs = modelDetail
    ? Array.from(new Set(modelDetail.configurations.map(c => c.num_gpus))).sort((a, b) => a - b)
    : [];

  const handleGPUToggle = (gpu: string) => {
    const newSet = new Set(selectedGPUs);
    if (newSet.has(gpu)) {
      newSet.delete(gpu);
    } else {
      newSet.add(gpu);
    }
    setSelectedGPUs(newSet);
  };

  const handleNumGPUsToggle = (numGPUs: number) => {
    const newSet = new Set(selectedNumGPUs);
    if (newSet.has(numGPUs)) {
      newSet.delete(numGPUs);
    } else {
      newSet.add(numGPUs);
    }
    setSelectedNumGPUs(newSet);
  };

  const handleSort = (key: string) => {
    if (sortKey === key) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortKey(null);
      }
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  const getSortIcon = (key: string) => {
    if (sortKey !== key) {
      return <span className="ml-1 text-gray-400">⇅</span>;
    }

    if (sortDirection === 'asc') {
      return <span className="ml-1 text-blue-600">↑</span>;
    } else if (sortDirection === 'desc') {
      return <span className="ml-1 text-blue-600">↓</span>;
    }

    return <span className="ml-1 text-gray-400">⇅</span>;
  };

  // Helper to get latency value from config
  const getLatencyValue = (config: AnyModelConfiguration): number => {
    if ('batch_latency_s' in config) return config.batch_latency_s;
    return (config as ModelConfiguration).median_itl_ms;
  };

  // Apply filters first, then sort
  const filteredConfigs = useMemo(() => {
    if (!modelDetail) return [];
    return modelDetail.configurations.filter(
      (config) =>
        selectedGPUs.has(config.gpu_model) && selectedNumGPUs.has(config.num_gpus)
    );
  }, [modelDetail, selectedGPUs, selectedNumGPUs]);

  // Compute Pareto optimal configurations from filtered configs
  const paretoOptimalConfigs = useMemo(() => {
    if (filteredConfigs.length === 0) return new Set<AnyModelConfiguration>();

    const paretoSet = new Set<AnyModelConfiguration>();
    for (const config of filteredConfigs) {
      const configEnergy = getEnergyValue(config) ?? Infinity;
      const configLatency = getLatencyValue(config);

      // A config is Pareto optimal if no other config dominates it
      // (i.e., no other config has both lower energy AND lower latency)
      const isDominated = filteredConfigs.some(other => {
        if (other === config) return false;
        const otherEnergy = getEnergyValue(other) ?? Infinity;
        const otherLatency = getLatencyValue(other);
        return otherEnergy < configEnergy && otherLatency < configLatency;
      });

      if (!isDominated) {
        paretoSet.add(config);
      }
    }
    return paretoSet;
  }, [filteredConfigs]);

  const sortedConfigs = [...filteredConfigs].sort((a, b) => {
    if (!sortKey || !sortDirection) return 0;

    const aVal = (a as any)[sortKey];
    const bVal = (b as any)[sortKey];

    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDirection === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }

    return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
  });

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-7xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex justify-between items-center z-10">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              {modelId}
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Task: {task}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {onAddToComparison && (
              <button
                onClick={() => onAddToComparison(modelId)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Add to comparison
              </button>
            )}
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold leading-none"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <p className="mt-4 text-gray-600 dark:text-gray-400">Loading model details...</p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          {modelDetail && (
            <div className="space-y-8">
              {/* Model Info */}
              {'total_params_billions' in modelDetail && (
                <div className={`grid grid-cols-2 ${isDiffusion ? '' : 'md:grid-cols-4'} gap-4 bg-gray-50 dark:bg-gray-900 rounded-lg p-4`}>
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{isDiffusion ? 'Parameters' : 'Total Parameters'}</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {modelDetail.total_params_billions < 1
                        ? `${(modelDetail.total_params_billions * 1000).toFixed(0)}M`
                        : `${modelDetail.total_params_billions.toFixed(1)}B`}
                    </p>
                  </div>
                  {!isDiffusion && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Active Parameters</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {modelDetail.activated_params_billions < 1
                          ? `${(modelDetail.activated_params_billions * 1000).toFixed(0)}M`
                          : `${modelDetail.activated_params_billions.toFixed(1)}B`}
                      </p>
                    </div>
                  )}
                  {!isDiffusion && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Architecture</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {(modelDetail as ModelDetail).architecture}
                      </p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">Weight Precision</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {modelDetail.weight_precision}
                    </p>
                  </div>
                </div>
              )}

              {/* Time-Energy Tradeoff and Energy/Response Distribution */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                  Time-Energy Tradeoff
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Each point is a hardware configuration for this model. Pareto frontier computed across these configurations.
                </p>
                <div className={`grid grid-cols-1 ${isDiffusion ? '' : 'lg:grid-cols-2'} gap-6 items-stretch`}>
                  {/* Tradeoff Chart */}
                  <div className={`bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col ${isDiffusion ? 'mx-auto w-full max-w-4xl' : ''}`}>
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                      Model Configurations and the Pareto Frontier
                    </h4>
                    <div className="flex-1 min-h-0">
                      <TimeEnergyTradeoffChart
                        configurations={filteredConfigs}
                        task={task}
                        selectedPercentile={selectedPercentile}
                        onPercentileChange={setSelectedPercentile}
                        onHoverConfig={(config) => {
                          // Only update when hovering a config, not when hover ends
                          if (config) {
                            setDisplayedConfig(config as AnyModelConfiguration);
                          }
                        }}
                      />
                    </div>
                  </div>

                  {/* Energy/Response Distribution - only for LLM/MLLM */}
                  {!isDiffusion && 'output_length_distribution' in modelDetail && (
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col">
                      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                        Energy per Response Distribution
                      </h4>
                      <div className="flex-1 min-h-0">
                        <EnergyPerResponseChart
                          outputLengthDistribution={
                            (displayedConfig as ModelConfiguration)?.output_length_distribution || (modelDetail as ModelDetail).output_length_distribution
                          }
                          energyPerToken={(displayedConfig as ModelConfiguration)?.energy_per_token_joules || null}
                          maxEnergyPerResponse={maxEnergyPerResponse > 0 ? maxEnergyPerResponse : null}
                          configLabel={
                            displayedConfig
                              ? `${displayedConfig.num_gpus} × ${displayedConfig.gpu_model}${(displayedConfig as ModelConfiguration).max_num_seqs ? `, max batch size ${(displayedConfig as ModelConfiguration).max_num_seqs}` : ''}`
                              : undefined
                          }
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Configuration Comparison Table */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  All Configurations for this Model
                </h3>

                {/* Filters */}
                <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-3">
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      GPU Model
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {availableGPUs.map((gpu) => (
                        <label
                          key={gpu}
                          className="flex items-center space-x-2 px-3 py-1 bg-white dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <input
                            type="checkbox"
                            checked={selectedGPUs.has(gpu)}
                            onChange={() => handleGPUToggle(gpu)}
                            className="rounded border-gray-300 dark:border-gray-600"
                          />
                          <span className="text-sm text-gray-900 dark:text-gray-100">{gpu}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Number of GPUs
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {availableNumGPUs.map((numGPUs) => (
                        <label
                          key={numGPUs}
                          className="flex items-center space-x-2 px-3 py-1 bg-white dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <input
                            type="checkbox"
                            checked={selectedNumGPUs.has(numGPUs)}
                            onChange={() => handleNumGPUsToggle(numGPUs)}
                            className="rounded border-gray-300 dark:border-gray-600"
                          />
                          <span className="text-sm text-gray-900 dark:text-gray-100">{numGPUs}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Showing {sortedConfigs.length} of {modelDetail?.configurations.length || 0} configurations
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-900">
                      <tr>
                        <th
                          onClick={() => handleSort('gpu_model')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            GPU
                            {getSortIcon('gpu_model')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort('num_gpus')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            # GPUs
                            {getSortIcon('num_gpus')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort(isDiffusion ? 'batch_size' : 'max_num_seqs')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            {isDiffusion ? 'Batch Size' : 'Max Batch'}
                            {getSortIcon(isDiffusion ? 'batch_size' : 'max_num_seqs')}
                          </div>
                        </th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Parallelization</th>
                        <th
                          onClick={() => handleSort(defaultSortKey)}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            {task === 'text-to-image' ? 'Energy/Image (J)' : task === 'text-to-video' ? 'Energy/Video (J)' : 'Energy/Token (J)'}
                            {getSortIcon(defaultSortKey)}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort(isDiffusion ? 'batch_latency_s' : 'median_itl_ms')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            {isDiffusion ? 'Batch Latency (s)' : 'Median ITL (ms)'}
                            {getSortIcon(isDiffusion ? 'batch_latency_s' : 'median_itl_ms')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort(isDiffusion ? (task === 'text-to-image' ? 'throughput_images_per_sec' : 'throughput_videos_per_sec') : 'output_throughput_tokens_per_sec')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            {task === 'text-to-image' ? 'Throughput (img/s)' : task === 'text-to-video' ? 'Throughput (vid/s)' : 'Throughput (tok/s)'}
                            {getSortIcon(isDiffusion ? (task === 'text-to-image' ? 'throughput_images_per_sec' : 'throughput_videos_per_sec') : 'output_throughput_tokens_per_sec')}
                          </div>
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {sortedConfigs.map((config, idx) => {
                        // Highlight Pareto optimal configurations
                        const isParetoOptimal = paretoOptimalConfigs.has(config);

                        let parallelStr = '-';
                        if (config.num_gpus > 1) {
                          if (isDiffusion) {
                            const diffConfig = config as ImageModelConfiguration | VideoModelConfiguration;
                            if (diffConfig.parallelization) {
                              const parts = [];
                              if (diffConfig.parallelization.ulysses_degree > 1) {
                                parts.push(`U${diffConfig.parallelization.ulysses_degree}`);
                              }
                              if (diffConfig.parallelization.ring_degree > 1) {
                                parts.push(`R${diffConfig.parallelization.ring_degree}`);
                              }
                              parallelStr = parts.length > 0 ? parts.join('+') : '-';
                            }
                          } else {
                            const llmConfig = config as ModelConfiguration;
                            if (llmConfig.parallelization) {
                              const parallelization = [];
                              if (llmConfig.parallelization.tensor_parallel > 1) {
                                parallelization.push(`TP${llmConfig.parallelization.tensor_parallel}`);
                              }
                              if (llmConfig.parallelization.expert_parallel > 1) {
                                parallelization.push(`EP${llmConfig.parallelization.expert_parallel}`);
                              }
                              if (llmConfig.parallelization.data_parallel > 1) {
                                parallelization.push(`DP${llmConfig.parallelization.data_parallel}`);
                              }
                              parallelStr = parallelization.length > 0 ? parallelization.join('+') : `TP${config.num_gpus}`;
                            }
                          }
                        }

                        // Get cell values based on config type (with null checks for task transitions)
                        const batchCell = isDiffusion
                          ? (config as ImageModelConfiguration | VideoModelConfiguration).batch_size ?? '-'
                          : (config as ModelConfiguration).max_num_seqs || '-';
                        const energyValue = getEnergyValue(config);
                        const energyCell = energyValue != null ? energyValue.toFixed(isDiffusion ? 1 : 4) : '-';
                        const latencyValue = isDiffusion
                          ? (config as ImageModelConfiguration | VideoModelConfiguration).batch_latency_s
                          : (config as ModelConfiguration).median_itl_ms;
                        const latencyCell = latencyValue != null ? latencyValue.toFixed(isDiffusion ? 2 : 1) : '-';
                        const throughputValue = task === 'text-to-image'
                          ? (config as ImageModelConfiguration).throughput_images_per_sec
                          : task === 'text-to-video'
                            ? (config as VideoModelConfiguration).throughput_videos_per_sec
                            : (config as ModelConfiguration).output_throughput_tokens_per_sec;
                        const throughputCell = throughputValue != null
                          ? throughputValue.toFixed(task === 'text-to-image' ? 3 : task === 'text-to-video' ? 4 : 0)
                          : '-';

                        return (
                          <tr
                            key={idx}
                            className={isParetoOptimal ? 'bg-amber-50 dark:bg-amber-900/20' : ''}
                          >
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.gpu_model}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.num_gpus}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{batchCell}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{parallelStr}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{energyCell}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{latencyCell}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{throughputCell}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
