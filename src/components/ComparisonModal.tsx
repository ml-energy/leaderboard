import { useEffect, useState, useMemo } from 'react';
import { ModelDetail, ModelConfiguration, Distribution } from '../types';
import { TimeEnergyTradeoffChart, ITLPercentile } from './TimeEnergyTradeoffChart';
import { EnergyPerResponseChart } from './EnergyPerResponseChart';

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
  onAddModel: () => void;
  onRemoveModel: (modelId: string) => void;
}

interface CombinedConfiguration extends ModelConfiguration {
  model_id: string;
  nickname: string;
  total_params_billions: number;
  activated_params_billions: number;
  is_moe: boolean;
  weight_precision: string;
}

type SortDirection = 'asc' | 'desc' | null;

export function ComparisonModal({
  modelIds,
  task,
  isOpen,
  onClose,
  onAddModel,
  onRemoveModel,
}: ComparisonModalProps) {
  const [modelDetails, setModelDetails] = useState<Map<string, ModelDetail>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<string | null>('energy_per_token_joules');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [selectedNumGPUs, setSelectedNumGPUs] = useState<Set<number>>(new Set());
  const [displayedConfig, setDisplayedConfig] = useState<CombinedConfiguration | null>(null);
  const [selectedPercentile, setSelectedPercentile] = useState<ITLPercentile>('p50');

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
          const data: ModelDetail = await response.json();
          return [modelId, data] as [string, ModelDetail];
        });

        const results = await Promise.all(promises);
        setModelDetails(new Map(results));
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
      detail.configurations.forEach(c => {
        allGPUs.add(c.gpu_model);
        allNumGPUs.add(c.num_gpus);
      });
    });

    setSelectedGPUs(allGPUs);
    setSelectedNumGPUs(allNumGPUs);

    // Initialize displayed config to highest energy/token across all models
    const allConfigs = getCombinedConfigurations();
    if (allConfigs.length > 0) {
      const highestEnergyConfig = allConfigs.reduce((max, c) =>
        c.energy_per_token_joules > max.energy_per_token_joules ? c : max
      );
      setDisplayedConfig(highestEnergyConfig);
    }
  }, [modelDetails]);

  // Combine all configurations from all models
  const getCombinedConfigurations = (): CombinedConfiguration[] => {
    const combined: CombinedConfiguration[] = [];
    modelDetails.forEach((detail, modelId) => {
      detail.configurations.forEach(config => {
        combined.push({
          ...config,
          model_id: modelId,
          nickname: modelId.split('/').pop() || modelId,
          total_params_billions: detail.total_params_billions,
          activated_params_billions: detail.activated_params_billions,
          is_moe: detail.is_moe,
          weight_precision: detail.weight_precision,
        });
      });
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
  }, [modelDetails, modelIds, selectedGPUs, selectedNumGPUs]);

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

  // Calculate energy stats for histogram X-axis
  const { defaultEnergyPerToken, maxEnergyPerToken } = useMemo(() => {
    if (filteredConfigs.length === 0) return { defaultEnergyPerToken: 0, maxEnergyPerToken: 0 };
    const energies = filteredConfigs.map(c => c.energy_per_token_joules).sort((a, b) => a - b);
    const max = energies[energies.length - 1];
    const medianIdx = Math.floor(energies.length / 2);
    const median = energies.length % 2 === 0
      ? (energies[medianIdx - 1] + energies[medianIdx]) / 2
      : energies[medianIdx];
    return { defaultEnergyPerToken: median, maxEnergyPerToken: max };
  }, [filteredConfigs]);

  // Get available GPUs and num_gpus for filters
  const availableGPUs = useMemo(() => {
    const gpus = new Set<string>();
    modelDetails.forEach(detail => {
      detail.configurations.forEach(c => gpus.add(c.gpu_model));
    });
    return Array.from(gpus).sort();
  }, [modelDetails]);

  const availableNumGPUs = useMemo(() => {
    const numGPUs = new Set<number>();
    modelDetails.forEach(detail => {
      detail.configurations.forEach(c => numGPUs.add(c.num_gpus));
    });
    return Array.from(numGPUs).sort((a, b) => a - b);
  }, [modelDetails]);

  // Get output length distribution (use first model's distribution as reference)
  const outputLengthDistribution = useMemo((): Distribution | null => {
    if (displayedConfig) {
      return displayedConfig.output_length_distribution;
    }
    const firstDetail = modelDetails.values().next().value;
    return firstDetail?.output_length_distribution || null;
  }, [modelDetails, displayedConfig]);

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
          <div className="flex items-center gap-3">
            {modelIds.length < 8 && (
              <button
                onClick={onAddModel}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Add another model
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
                          {modelId.split('/').pop()}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {detail.total_params_billions.toFixed(1)}B params, {detail.is_moe ? 'MoE' : 'Dense'}, {detail.weight_precision}
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
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <TimeEnergyTradeoffChart
                      configurations={filteredConfigs}
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

                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                      Energy per Response Distribution
                    </h4>
                    {outputLengthDistribution && (
                      <EnergyPerResponseChart
                        outputLengthDistribution={outputLengthDistribution}
                        energyPerToken={displayedConfig?.energy_per_token_joules || null}
                        defaultEnergyPerToken={defaultEnergyPerToken}
                        maxEnergyPerToken={maxEnergyPerToken}
                        configLabel={
                          displayedConfig
                            ? `${displayedConfig.nickname} - ${displayedConfig.num_gpus} × ${displayedConfig.gpu_model}${displayedConfig.max_num_seqs ? `, batch ${displayedConfig.max_num_seqs}` : ''}`
                            : undefined
                        }
                      />
                    )}
                  </div>
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
                          onClick={() => handleSort('max_num_seqs')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Max Batch
                            {getSortIcon('max_num_seqs')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort('energy_per_token_joules')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Energy/Token (J)
                            {getSortIcon('energy_per_token_joules')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort('median_itl_ms')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Median ITL (ms)
                            {getSortIcon('median_itl_ms')}
                          </div>
                        </th>
                        <th
                          onClick={() => handleSort('output_throughput_tokens_per_sec')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Throughput (tok/s)
                            {getSortIcon('output_throughput_tokens_per_sec')}
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

                        return (
                          <tr key={idx}>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">
                              <div className="flex items-center gap-2">
                                <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color.hex }}></div>
                                <span className="truncate max-w-[150px]">{config.nickname}</span>
                              </div>
                            </td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.gpu_model}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.num_gpus}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.max_num_seqs || '-'}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.energy_per_token_joules.toFixed(4)}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.median_itl_ms.toFixed(1)}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.output_throughput_tokens_per_sec.toFixed(0)}</td>
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
