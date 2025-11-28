import { useEffect, useState, useMemo } from 'react';
import { ModelDetail, ModelConfiguration, Distribution, DiffusionModelDetail, ImageModelConfiguration, VideoModelConfiguration } from '../types';
import { TimeEnergyTradeoffChart, ITLPercentile } from './TimeEnergyTradeoffChart';
import { EnergyPerResponseChart } from './EnergyPerResponseChart';
import { isDiffusionTask } from '../config/tasks';

// Color palette for models
export const MODEL_COLORS = [
  { bg: 'bg-blue-100 dark:bg-blue-900/40', text: 'text-blue-600 dark:text-blue-400', hex: '#3b82f6' },
  { bg: 'bg-emerald-100 dark:bg-emerald-900/40', text: 'text-emerald-600 dark:text-emerald-400', hex: '#10b981' },
  { bg: 'bg-amber-100 dark:bg-amber-900/40', text: 'text-amber-600 dark:text-amber-400', hex: '#f59e0b' },
  { bg: 'bg-red-100 dark:bg-red-900/40', text: 'text-red-600 dark:text-red-400', hex: '#ef4444' },
  { bg: 'bg-violet-100 dark:bg-violet-900/40', text: 'text-violet-600 dark:text-violet-400', hex: '#8b5cf6' },
  { bg: 'bg-pink-100 dark:bg-pink-900/40', text: 'text-pink-600 dark:text-pink-400', hex: '#ec4899' },
  { bg: 'bg-teal-100 dark:bg-teal-900/40', text: 'text-teal-600 dark:text-teal-400', hex: '#14b8a6' },
  { bg: 'bg-orange-100 dark:bg-orange-900/40', text: 'text-orange-600 dark:text-orange-400', hex: '#f97316' },
];

interface ComparisonModalProps {
  modelIds: string[];
  task: string;
  isOpen: boolean;
  onClose: () => void;
  onRemoveModel: (modelId: string) => void;
  modelNicknames?: Record<string, string>;
}

// Combined configuration for LLM/MLLM
interface LLMCombinedConfiguration extends ModelConfiguration {
  model_id: string;
  nickname: string;
  total_params_billions: number;
  activated_params_billions: number;
  architecture: string;
  weight_precision: string;
}

// Combined configuration for diffusion (flat fields to match ImageConfiguration/VideoConfiguration)
interface DiffusionCombinedConfiguration {
  model_id: string;
  nickname: string;
  gpu_model: string;
  num_gpus: number;
  total_params_billions: number;
  activated_params_billions: number;
  weight_precision: string;
  batch_size: number;
  batch_latency_s: number;
  inference_steps: number;
  ulysses_degree: number;
  ring_degree: number;
  // Image-specific
  energy_per_image_joules?: number;
  throughput_images_per_sec?: number;
  image_height?: number;
  image_width?: number;
  // Video-specific
  energy_per_video_joules?: number;
  throughput_videos_per_sec?: number;
  video_height?: number;
  video_width?: number;
}

type CombinedConfiguration = LLMCombinedConfiguration | DiffusionCombinedConfiguration;
type AnyModelDetail = ModelDetail | DiffusionModelDetail;

type SortDirection = 'asc' | 'desc' | null;

export function ComparisonModal({
  modelIds,
  task,
  isOpen,
  onClose,
  onRemoveModel,
  modelNicknames = {},
}: ComparisonModalProps) {
  const isDiffusion = isDiffusionTask(task);
  const [modelDetails, setModelDetails] = useState<Map<string, AnyModelDetail>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const defaultSortKey = task === 'text-to-image' ? 'energy_per_image_joules' : task === 'text-to-video' ? 'energy_per_video_joules' : 'energy_per_token_joules';
  const [sortKey, setSortKey] = useState<string | null>(defaultSortKey);
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [selectedNumGPUs, setSelectedNumGPUs] = useState<Set<number>>(new Set());
  const [displayedConfig, setDisplayedConfig] = useState<CombinedConfiguration | null>(null);
  const [selectedPercentile, setSelectedPercentile] = useState<ITLPercentile>('p50');

  // Helper to get energy value
  const getEnergyValue = (config: CombinedConfiguration): number | undefined => {
    if ('energy_per_image_joules' in config) return config.energy_per_image_joules;
    if ('energy_per_video_joules' in config) return config.energy_per_video_joules;
    if ('energy_per_token_joules' in config) return (config as LLMCombinedConfiguration).energy_per_token_joules;
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

  // Load all model details
  useEffect(() => {
    if (!isOpen || modelIds.length === 0) return;

    const loadAllModels = async () => {
      setLoading(true);
      setError(null);
      try {
        const promises = modelIds.map(async (modelId) => {
          const detailFilePath = `models/${modelId.replace('/', '__')}__${task}.json`;
          const response = await fetch(`${import.meta.env.BASE_URL}data/${detailFilePath}`);
          if (!response.ok) {
            throw new Error(`Failed to load ${modelId}: ${response.statusText}`);
          }
          const data: AnyModelDetail = await response.json();
          return [modelId, data] as [string, AnyModelDetail];
        });

        const results = await Promise.allSettled(promises);
        const successfulResults = results
          .filter((r): r is PromiseFulfilledResult<[string, AnyModelDetail]> => r.status === 'fulfilled')
          .map(r => r.value);

        if (successfulResults.length === 0) {
          throw new Error('Failed to load any model details');
        }
        setModelDetails(new Map(successfulResults));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load model details');
      } finally {
        setLoading(false);
      }
    };

    loadAllModels();
  }, [isOpen, modelIds, task]);

  // Initialize filters when model details load
  useEffect(() => {
    if (modelDetails.size === 0) return;

    const allGPUs = new Set<string>();
    const allNumGPUs = new Set<number>();

    modelDetails.forEach(detail => {
      detail.configurations.forEach((c: any) => {
        allGPUs.add(c.gpu_model);
        allNumGPUs.add(c.num_gpus);
      });
    });

    setSelectedGPUs(allGPUs);
    setSelectedNumGPUs(allNumGPUs);

    // Initialize displayed config to highest energy across all models
    const allConfigs = getCombinedConfigurations();
    if (allConfigs.length > 0) {
      const highestEnergyConfig = allConfigs.reduce((max, c) =>
        (getEnergyValue(c) ?? 0) > (getEnergyValue(max) ?? 0) ? c : max
      );
      setDisplayedConfig(highestEnergyConfig);
    }
  }, [modelDetails]);

  // Combine all configurations from all models
  const getCombinedConfigurations = (): CombinedConfiguration[] => {
    const combined: CombinedConfiguration[] = [];
    modelDetails.forEach((detail, modelId) => {
      if (isDiffusion) {
        const diffDetail = detail as DiffusionModelDetail;
        diffDetail.configurations.forEach((config: ImageModelConfiguration | VideoModelConfiguration) => {
          // Flatten parallelization fields to match ImageConfiguration/VideoConfiguration format
          const { parallelization, ...restConfig } = config;
          combined.push({
            ...restConfig,
            model_id: modelId,
            nickname: modelNicknames[modelId] || diffDetail.nickname || modelId.split('/').pop() || modelId,
            total_params_billions: diffDetail.total_params_billions,
            activated_params_billions: diffDetail.activated_params_billions,
            weight_precision: diffDetail.weight_precision,
            ulysses_degree: parallelization.ulysses_degree,
            ring_degree: parallelization.ring_degree,
          } as DiffusionCombinedConfiguration);
        });
      } else {
        const llmDetail = detail as ModelDetail;
        llmDetail.configurations.forEach(config => {
          combined.push({
            ...config,
            model_id: modelId,
            nickname: modelNicknames[modelId] || modelId.split('/').pop() || modelId,
            total_params_billions: llmDetail.total_params_billions,
            activated_params_billions: llmDetail.activated_params_billions,
            architecture: llmDetail.architecture,
            weight_precision: llmDetail.weight_precision,
          } as LLMCombinedConfiguration);
        });
      }
    });
    return combined;
  };

  // Filter and sort configurations
  const filteredConfigs = useMemo(() => {
    return getCombinedConfigurations().filter(
      config =>
        modelIds.includes(config.model_id) &&
        selectedGPUs.has(config.gpu_model) &&
        selectedNumGPUs.has(config.num_gpus)
    );
  }, [modelDetails, modelIds, selectedGPUs, selectedNumGPUs, modelNicknames, isDiffusion]);

  const sortedConfigs = useMemo(() => {
    return [...filteredConfigs].sort((a, b) => {
      if (!sortKey || !sortDirection) return 0;

      let aVal = (a as any)[sortKey];
      let bVal = (b as any)[sortKey];

      // Handle secondary sort by model_id
      if (aVal === bVal && sortKey !== 'model_id') {
        aVal = a.model_id;
        bVal = b.model_id;
        return aVal.localeCompare(bVal);
      }

      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });
  }, [filteredConfigs, sortKey, sortDirection]);

  // Calculate actual max energy by looking at each config's distribution × energy_per_token
  const maxEnergyPerResponse = useMemo(() => {
    if (isDiffusion || filteredConfigs.length === 0) return 0;
    let maxEnergy = 0;
    for (const config of filteredConfigs) {
      const llmConfig = config as LLMCombinedConfiguration;
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
    return maxEnergy;
  }, [filteredConfigs, isDiffusion]);

  // Get available GPUs and num_gpus for filters
  const availableGPUs = useMemo(() => {
    const gpus = new Set<string>();
    modelDetails.forEach(detail => {
      detail.configurations.forEach((c: any) => gpus.add(c.gpu_model));
    });
    return Array.from(gpus).sort();
  }, [modelDetails]);

  const availableNumGPUs = useMemo(() => {
    const numGPUs = new Set<number>();
    modelDetails.forEach(detail => {
      detail.configurations.forEach((c: any) => numGPUs.add(c.num_gpus));
    });
    return Array.from(numGPUs).sort((a, b) => a - b);
  }, [modelDetails]);

  // Get output length distribution (only for LLM/MLLM)
  const outputLengthDistribution = useMemo((): Distribution | null => {
    if (isDiffusion) return null;
    if (displayedConfig && 'output_length_distribution' in displayedConfig) {
      return (displayedConfig as LLMCombinedConfiguration).output_length_distribution;
    }
    const firstDetail = modelDetails.values().next().value as ModelDetail | undefined;
    return firstDetail?.output_length_distribution || null;
  }, [modelDetails, displayedConfig, isDiffusion]);

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
              Comparing {modelIds.length} models
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Task: {task}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold leading-none"
          >
            ×
          </button>
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

          {modelDetails.size > 0 && !loading && (
            <div className="space-y-8">
              {/* Model Info Cards */}
              <div className="flex flex-wrap gap-4">
                {modelIds.map((modelId, index) => {
                  const detail = modelDetails.get(modelId);
                  const color = MODEL_COLORS[index % MODEL_COLORS.length];
                  if (!detail) return null;

                  return (
                    <div
                      key={modelId}
                      className={`flex-1 min-w-[200px] ${color.bg} rounded-lg p-4 relative`}
                    >
                      {modelIds.length > 2 && (
                        <button
                          onClick={() => onRemoveModel(modelId)}
                          className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                        >
                          ×
                        </button>
                      )}
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: color.hex }}></div>
                        <h3 className="font-semibold text-gray-900 dark:text-white truncate">
                          {isDiffusion
                            ? (detail as DiffusionModelDetail).nickname || modelId.split('/').pop()
                            : modelNicknames[modelId] || modelId.split('/').pop()}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {detail.total_params_billions < 1
                          ? `${(detail.total_params_billions * 1000).toFixed(0)}M`
                          : `${detail.total_params_billions.toFixed(1)}B`} params
                        {!isDiffusion && `, ${(detail as ModelDetail).architecture}`}
                        , {detail.weight_precision}
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Time-Energy Tradeoff */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Time-Energy Tradeoff
                </h3>
                <div className={`grid grid-cols-1 ${isDiffusion ? '' : 'lg:grid-cols-2'} gap-6 items-stretch`}>
                  <div className={`bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col ${isDiffusion ? 'mx-auto w-full max-w-4xl' : ''}`}>
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                      Model Configurations and the Pareto Frontier
                    </h4>
                    <div className="flex-1 min-h-0">
                      <TimeEnergyTradeoffChart
                        configurations={filteredConfigs as any[]}
                        task={task}
                        selectedPercentile={selectedPercentile}
                        onPercentileChange={setSelectedPercentile}
                        onHoverConfig={(config) => {
                          if (config) {
                            setDisplayedConfig(config as CombinedConfiguration);
                          }
                        }}
                        colorByModel={true}
                      />
                    </div>
                  </div>

                  {!isDiffusion && (
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col">
                      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                        Energy per Response Distribution
                      </h4>
                      <div className="flex-1 min-h-0">
                        {outputLengthDistribution && (
                          <EnergyPerResponseChart
                            outputLengthDistribution={outputLengthDistribution}
                            energyPerToken={(displayedConfig as LLMCombinedConfiguration)?.energy_per_token_joules ?? null}
                            maxEnergyPerResponse={maxEnergyPerResponse}
                            configLabel={
                              displayedConfig
                                ? `${displayedConfig.nickname} - ${displayedConfig.num_gpus} × ${displayedConfig.gpu_model}${(displayedConfig as LLMCombinedConfiguration).max_num_seqs ? `, batch ${(displayedConfig as LLMCombinedConfiguration).max_num_seqs}` : ''}`
                                : undefined
                            }
                          />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Filters */}
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-3">
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
                  Showing {sortedConfigs.length} configurations across {modelIds.length} models
                </div>
              </div>

              {/* Configuration Table */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  All Configurations
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-900">
                      <tr>
                        <th
                          onClick={() => handleSort('model_id')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Model
                            {getSortIcon('model_id')}
                          </div>
                        </th>
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
                        const modelIndex = modelIds.indexOf(config.model_id);
                        // Skip configs for models that were removed
                        if (modelIndex === -1) return null;
                        const color = MODEL_COLORS[modelIndex % MODEL_COLORS.length];

                        // Determine if row is hovered based on config type
                        const isHovered = isDiffusion
                          ? displayedConfig &&
                            config.model_id === displayedConfig.model_id &&
                            config.gpu_model === displayedConfig.gpu_model &&
                            config.num_gpus === displayedConfig.num_gpus &&
                            (config as DiffusionCombinedConfiguration).batch_size === (displayedConfig as DiffusionCombinedConfiguration).batch_size &&
                            (config as DiffusionCombinedConfiguration).ulysses_degree === (displayedConfig as DiffusionCombinedConfiguration).ulysses_degree &&
                            (config as DiffusionCombinedConfiguration).ring_degree === (displayedConfig as DiffusionCombinedConfiguration).ring_degree
                          : displayedConfig &&
                            config.model_id === displayedConfig.model_id &&
                            config.gpu_model === displayedConfig.gpu_model &&
                            config.num_gpus === displayedConfig.num_gpus &&
                            (config as LLMCombinedConfiguration).max_num_seqs === (displayedConfig as LLMCombinedConfiguration).max_num_seqs;

                        // Get cell values based on config type (with null checks)
                        const batchCell = isDiffusion
                          ? (config as DiffusionCombinedConfiguration).batch_size ?? '-'
                          : (config as LLMCombinedConfiguration).max_num_seqs || '-';
                        const energyValue = getEnergyValue(config);
                        const energyCell = energyValue != null ? energyValue.toFixed(isDiffusion ? 1 : 4) : '-';
                        const latencyValue = isDiffusion
                          ? (config as DiffusionCombinedConfiguration).batch_latency_s
                          : (config as LLMCombinedConfiguration).median_itl_ms;
                        const latencyCell = latencyValue != null ? latencyValue.toFixed(isDiffusion ? 2 : 1) : '-';
                        const throughputValue = task === 'text-to-image'
                          ? (config as DiffusionCombinedConfiguration).throughput_images_per_sec
                          : task === 'text-to-video'
                            ? (config as DiffusionCombinedConfiguration).throughput_videos_per_sec
                            : (config as LLMCombinedConfiguration).output_throughput_tokens_per_sec;
                        const throughputCell = throughputValue != null
                          ? throughputValue.toFixed(task === 'text-to-image' ? 3 : task === 'text-to-video' ? 4 : 0)
                          : '-';

                        return (
                          <tr
                            key={idx}
                            className={isHovered ? 'bg-blue-50 dark:bg-blue-900/30' : ''}
                          >
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                              <div className="flex items-center gap-2">
                                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color.hex }}></div>
                                <span>{config.nickname}</span>
                              </div>
                            </td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.gpu_model}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.num_gpus}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{batchCell}</td>
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
