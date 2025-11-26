import { useEffect, useState, useMemo } from 'react';
import { ModelDetail, ModelConfiguration } from '../types';
import { TimeEnergyTradeoffChart, ITLPercentile } from './TimeEnergyTradeoffChart';
import { EnergyPerResponseChart } from './EnergyPerResponseChart';

interface ModelDetailModalProps {
  modelId: string;
  task: string;
  isOpen: boolean;
  onClose: () => void;
  onAddToComparison?: (modelId: string) => void;
  currentConfig?: {
    gpu_model: string;
    num_gpus: number;
    max_num_seqs: number | null;
  };
}

type SortDirection = 'asc' | 'desc' | null;

export function ModelDetailModal({
  modelId,
  task,
  isOpen,
  onClose,
  onAddToComparison,
  currentConfig: _currentConfig,
}: ModelDetailModalProps) {
  const [modelDetail, setModelDetail] = useState<ModelDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<string | null>('energy_per_token_joules');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [selectedNumGPUs, setSelectedNumGPUs] = useState<Set<number>>(new Set());
  const [displayedConfig, setDisplayedConfig] = useState<ModelConfiguration | null>(null);
  const [selectedPercentile, setSelectedPercentile] = useState<ITLPercentile>('p50');

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

  // Initialize filters when model detail loads
  useEffect(() => {
    if (modelDetail) {
      const gpuModels = new Set(modelDetail.configurations.map(c => c.gpu_model));
      const numGPUs = new Set(modelDetail.configurations.map(c => c.num_gpus));
      setSelectedGPUs(gpuModels);
      setSelectedNumGPUs(numGPUs);

      // Initialize displayed config to highest energy/token
      if (modelDetail.configurations.length > 0) {
        const highestEnergyConfig = modelDetail.configurations.reduce((max, c) =>
          c.energy_per_token_joules > max.energy_per_token_joules ? c : max
        );
        setDisplayedConfig(highestEnergyConfig);
      }
    }
  }, [modelDetail]);

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

  // Apply filters first, then sort
  const filteredConfigs = modelDetail
    ? modelDetail.configurations.filter(
        (config) =>
          selectedGPUs.has(config.gpu_model) && selectedNumGPUs.has(config.num_gpus)
      )
    : [];

  // Calculate energy per token statistics for X-axis scaling
  const { defaultEnergyPerToken, maxEnergyPerToken } = useMemo(() => {
    if (filteredConfigs.length === 0) return { defaultEnergyPerToken: 0, maxEnergyPerToken: 0 };
    const energies = filteredConfigs.map(c => c.energy_per_token_joules).sort((a, b) => a - b);
    const max = energies[energies.length - 1];
    // Use median as the default X-axis range
    const medianIdx = Math.floor(energies.length / 2);
    const median = energies.length % 2 === 0
      ? (energies[medianIdx - 1] + energies[medianIdx]) / 2
      : energies[medianIdx];
    return { defaultEnergyPerToken: median, maxEnergyPerToken: max };
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
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Total Parameters</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {modelDetail.total_params_billions.toFixed(1)}B
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Active Parameters</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {modelDetail.activated_params_billions.toFixed(1)}B
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Architecture</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {modelDetail.architecture}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Weight Precision</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {modelDetail.weight_precision}
                  </p>
                </div>
              </div>

              {/* Time-Energy Tradeoff and Energy/Response Distribution */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                  Time-Energy Tradeoff
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Each point is a hardware configuration for this model. Pareto frontier computed across these configurations.
                </p>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
                  {/* Tradeoff Chart */}
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col">
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                      Model Configurations and the Pareto Frontier
                    </h4>
                    <div className="flex-1 min-h-0">
                      <TimeEnergyTradeoffChart
                        configurations={filteredConfigs}
                        selectedPercentile={selectedPercentile}
                        onPercentileChange={setSelectedPercentile}
                        onHoverConfig={(config) => {
                          // Only update when hovering a config, not when hover ends
                          if (config) {
                            setDisplayedConfig(config as ModelConfiguration);
                          }
                        }}
                      />
                    </div>
                  </div>

                  {/* Energy/Response Distribution */}
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 flex flex-col">
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
                      Energy per Response Distribution
                    </h4>
                    <div className="flex-1 min-h-0">
                      <EnergyPerResponseChart
                        outputLengthDistribution={
                          displayedConfig?.output_length_distribution || modelDetail.output_length_distribution
                        }
                        energyPerToken={displayedConfig?.energy_per_token_joules || null}
                        defaultEnergyPerToken={defaultEnergyPerToken}
                        maxEnergyPerToken={maxEnergyPerToken}
                        configLabel={
                          displayedConfig
                            ? `${displayedConfig.num_gpus} × ${displayedConfig.gpu_model}${displayedConfig.max_num_seqs ? `, max batch size ${displayedConfig.max_num_seqs}` : ''}`
                            : undefined
                        }
                      />
                    </div>
                  </div>
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
                          onClick={() => handleSort('max_num_seqs')}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <div className="flex items-center">
                            Max Batch
                            {getSortIcon('max_num_seqs')}
                          </div>
                        </th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Parallelization</th>
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
                        const isHovered =
                          displayedConfig &&
                          config.gpu_model === displayedConfig.gpu_model &&
                          config.num_gpus === displayedConfig.num_gpus &&
                          config.max_num_seqs === displayedConfig.max_num_seqs;

                        let parallelStr = 'N/A';
                        if (config.num_gpus > 1) {
                          const parallelization = [];
                          if (config.parallelization.tensor_parallel > 1) {
                            parallelization.push(`TP${config.parallelization.tensor_parallel}`);
                          }
                          if (config.parallelization.expert_parallel > 1) {
                            parallelization.push(`EP${config.parallelization.expert_parallel}`);
                          }
                          if (config.parallelization.data_parallel > 1) {
                            parallelization.push(`DP${config.parallelization.data_parallel}`);
                          }
                          // Default to TP{num_gpus} if no parallelization detected for multi-GPU
                          parallelStr = parallelization.length > 0 ? parallelization.join('+') : `TP${config.num_gpus}`;
                        }

                        return (
                          <tr
                            key={idx}
                            className={isHovered ? 'bg-amber-50 dark:bg-amber-900/20' : ''}
                          >
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.gpu_model}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.num_gpus}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{config.max_num_seqs || '-'}</td>
                            <td className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100">{parallelStr}</td>
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
