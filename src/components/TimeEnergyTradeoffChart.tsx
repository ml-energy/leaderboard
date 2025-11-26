import { useMemo, useEffect, useRef, useState } from 'react';
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
  TooltipProps,
} from 'recharts';
import { Configuration, ModelConfiguration } from '../types';

export type ITLPercentile = 'p50' | 'p90' | 'p95' | 'p99';

interface TimeEnergyTradeoffChartProps {
  configurations: (Configuration | ModelConfiguration)[];
  selectedPercentile: ITLPercentile;
  onPercentileChange: (p: ITLPercentile) => void;
  onHoverConfig: (config: Configuration | ModelConfiguration | null) => void;
  colorByModel?: boolean;
  onLegendClick?: (modelId: string) => void;
}

// Model colors for multi-model view (synced with ComparisonModal)
const MODEL_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
];

function getITL(config: Configuration | ModelConfiguration, percentile: ITLPercentile): number | null {
  switch (percentile) {
    case 'p50':
      return config.median_itl_ms;
    case 'p90':
      return config.p90_itl_ms ?? null;
    case 'p95':
      return config.p95_itl_ms;
    case 'p99':
      return config.p99_itl_ms ?? null;
  }
}

function getParetoFrontier<T extends Configuration | ModelConfiguration>(
  configs: T[],
  percentile: ITLPercentile
): T[] {
  // Filter configs that have valid ITL data for this percentile
  const validConfigs = configs.filter(c => getITL(c, percentile) !== null);

  return validConfigs.filter(config => {
    const configITL = getITL(config, percentile)!;
    // Pareto-optimal if no other config dominates (lower energy AND lower latency)
    return !validConfigs.some(other => {
      const otherITL = getITL(other, percentile)!;
      return other.energy_per_token_joules < config.energy_per_token_joules &&
        otherITL < configITL;
    });
  });
}

interface ChartDataPoint {
  x: number;
  y: number;
  config: Configuration | ModelConfiguration;
  isPareto: boolean;
  modelColor: string;
}

interface TooltipCaptureProps extends TooltipProps<number, string> {
  onActiveChange: (data: ChartDataPoint | null) => void;
}

// Invisible component that captures active data point from Recharts
function TooltipCapture({ active, payload, onActiveChange }: TooltipCaptureProps) {
  useEffect(() => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload as ChartDataPoint;
      onActiveChange(data);
    } else {
      onActiveChange(null);
    }
  }, [active, payload, onActiveChange]);

  return null; // Render nothing - we handle display separately
}

// Standalone tooltip component rendered outside Recharts
function FloatingTooltip({
  data,
  mouseX,
  mouseY,
}: {
  data: ChartDataPoint;
  mouseX: number;
  mouseY: number;
}) {
  const config = data.config;
  const hasModelId = 'model_id' in config;

  return (
    <div
      className="absolute bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 text-sm pointer-events-none z-50 whitespace-nowrap"
      style={{
        left: mouseX + 15,
        top: mouseY,
        transform: 'translateY(-100%)',
      }}
    >
      {hasModelId && (
        <p className="font-semibold text-gray-900 dark:text-white">
          {(config as Configuration).nickname || (config as Configuration).model_id}
        </p>
      )}
      <p className="text-gray-600 dark:text-gray-300">
        {config.num_gpus} × {config.gpu_model}
        {config.max_num_seqs && `, batch ${config.max_num_seqs}`}
      </p>
      <div className="mt-2 space-y-1 text-gray-700 dark:text-gray-200">
        <p>Energy: <span className="font-medium">{config.energy_per_token_joules.toFixed(4)} J/tok</span></p>
        <p>Median ITL: <span className="font-medium">{config.median_itl_ms.toFixed(1)} ms</span></p>
        <p>Throughput: <span className="font-medium">{config.output_throughput_tokens_per_sec.toFixed(0)} tok/s</span></p>
      </div>
      {data.isPareto && (
        <p className="mt-2 text-amber-600 dark:text-amber-400 font-medium text-xs">
          Pareto optimal
        </p>
      )}
    </div>
  );
}

export function TimeEnergyTradeoffChart({
  configurations,
  selectedPercentile,
  onPercentileChange,
  onHoverConfig,
  colorByModel = false,
  onLegendClick,
}: TimeEnergyTradeoffChartProps) {
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null);
  const [activeData, setActiveData] = useState<ChartDataPoint | null>(null);
  const [hoveredLegendModel, setHoveredLegendModel] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync hovered config with parent
  useEffect(() => {
    onHoverConfig(activeData?.config ?? null);
  }, [activeData, onHoverConfig]);

  // Get unique model IDs for color assignment
  const modelIds = useMemo(() => {
    const ids = new Set<string>();
    configurations.forEach(c => {
      if ('model_id' in c) {
        ids.add((c as Configuration).model_id);
      }
    });
    return Array.from(ids);
  }, [configurations]);

  // Map from modelId to nickname
  const modelNicknames = useMemo(() => {
    const map = new Map<string, string>();
    configurations.forEach(c => {
      if ('model_id' in c) {
        const config = c as Configuration;
        if (!map.has(config.model_id)) {
          map.set(config.model_id, config.nickname || config.model_id);
        }
      }
    });
    return map;
  }, [configurations]);

  const getModelColor = (modelId: string): string => {
    const index = modelIds.indexOf(modelId);
    return MODEL_COLORS[index % MODEL_COLORS.length];
  };

  // Calculate Pareto frontier - either global or per-model
  const paretoConfigs = useMemo(
    () => getParetoFrontier(configurations, selectedPercentile),
    [configurations, selectedPercentile]
  );

  // Per-model Pareto frontiers for comparison mode
  const perModelParetoConfigs = useMemo(() => {
    if (!colorByModel) return new Map<string, (Configuration | ModelConfiguration)[]>();

    const result = new Map<string, (Configuration | ModelConfiguration)[]>();
    modelIds.forEach(modelId => {
      const modelConfigs = configurations.filter(c =>
        'model_id' in c && (c as Configuration).model_id === modelId
      );
      result.set(modelId, getParetoFrontier(modelConfigs, selectedPercentile));
    });
    return result;
  }, [colorByModel, configurations, selectedPercentile, modelIds]);

  // Compute fixed X-axis max using p99 values (the largest ITL percentile)
  // Round up to a multiple of a nice tick interval for even grid spacing
  const xAxisMax = useMemo(() => {
    const p99Values = configurations
      .map(config => config.p99_itl_ms)
      .filter((v): v is number => v != null && v > 0);
    if (p99Values.length === 0) return undefined;
    const max = Math.max(...p99Values);
    // Choose a nice tick interval based on magnitude
    const magnitude = Math.pow(10, Math.floor(Math.log10(max)));
    const normalized = max / magnitude;
    let tickInterval: number;
    if (normalized <= 2) tickInterval = magnitude * 0.5;
    else if (normalized <= 5) tickInterval = magnitude;
    else tickInterval = magnitude * 2;
    // Round max up to next multiple of tick interval
    return Math.ceil(max / tickInterval) * tickInterval;
  }, [configurations]);

  // Transform data for chart (filter out configs without valid ITL data)
  const chartData: ChartDataPoint[] = useMemo(() => {
    return configurations
      .filter(config => getITL(config, selectedPercentile) !== null)
      .map(config => {
        const modelId = 'model_id' in config ? (config as Configuration).model_id : '';
        // Use per-model Pareto in comparison mode, global Pareto otherwise
        const isPareto = colorByModel
          ? (perModelParetoConfigs.get(modelId)?.includes(config) ?? false)
          : paretoConfigs.includes(config);
        return {
          x: getITL(config, selectedPercentile)!,
          y: config.energy_per_token_joules,
          config,
          isPareto,
          modelColor: colorByModel ? getModelColor(modelId) : '#3b82f6',
        };
      });
  }, [configurations, selectedPercentile, paretoConfigs, perModelParetoConfigs, colorByModel, modelIds]);

  // Sort Pareto points by X for line drawing (global or per-model)
  const paretoLineData = useMemo(() => {
    return chartData
      .filter(d => d.isPareto)
      .sort((a, b) => a.x - b.x);
  }, [chartData]);

  // Per-model Pareto line data for comparison mode
  const perModelParetoLineData = useMemo(() => {
    if (!colorByModel) return new Map<string, ChartDataPoint[]>();

    const result = new Map<string, ChartDataPoint[]>();
    modelIds.forEach(modelId => {
      const modelParetoPoints = chartData
        .filter(d => {
          const dModelId = 'model_id' in d.config ? (d.config as Configuration).model_id : '';
          return d.isPareto && dModelId === modelId;
        })
        .sort((a, b) => a.x - b.x);
      result.set(modelId, modelParetoPoints);
    });
    return result;
  }, [colorByModel, chartData, modelIds]);

  const percentileOptions: { value: ITLPercentile; label: string }[] = [
    { value: 'p50', label: 'P50 (Median)' },
    { value: 'p90', label: 'P90' },
    { value: 'p95', label: 'P95' },
    { value: 'p99', label: 'P99' },
  ];

  return (
    <div>
      {/* Percentile selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {percentileOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => onPercentileChange(opt.value)}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${
              selectedPercentile === opt.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-[400px] bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            No data available for {selectedPercentile.toUpperCase()} percentile. Data regeneration required.
          </p>
        </div>
      ) : (
        <div
          ref={containerRef}
          className="relative overflow-visible"
          onMouseMove={(e) => {
            if (containerRef.current) {
              const rect = containerRef.current.getBoundingClientRect();
              setMousePosition({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
              });
            }
          }}
          onMouseLeave={() => {
            setMousePosition(null);
            setActiveData(null);
          }}
        >
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart margin={{ top: 20, right: 20, bottom: 60, left: 70 }}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-50" />
              <XAxis
                dataKey="x"
                type="number"
                domain={[0, xAxisMax ?? 'auto']}
                name="ITL"
                label={{
                  value: `Inter-Token Latency (${selectedPercentile.toUpperCase()}) [ms]`,
                  position: 'insideBottom',
                  offset: -10,
                  className: 'fill-gray-600 dark:fill-gray-400',
                }}
                tick={{ fontSize: 13 }}
              />
              <YAxis
                dataKey="y"
                type="number"
                domain={[0, 'auto']}
                name="Energy"
                label={{
                  value: 'Energy per Token [J]',
                  angle: -90,
                  position: 'insideLeft',
                  dx: -15,
                  dy: 60,
                  className: 'fill-gray-600 dark:fill-gray-400',
                }}
                tick={{ fontSize: 13 }}
                tickFormatter={(value: number) => value.toFixed(3)}
              />
              <Tooltip
                content={<TooltipCapture onActiveChange={setActiveData} />}
              />

              {/* Pareto frontier lines - per-model or global */}
              {colorByModel ? (
                // Per-model Pareto frontier lines (skip grayed ones entirely)
                modelIds
                  .filter(modelId => hoveredLegendModel === null || hoveredLegendModel === modelId)
                  .map(modelId => {
                    const lineData = perModelParetoLineData.get(modelId) || [];
                    return (
                      <Line
                        key={`pareto-${modelId}`}
                        data={lineData}
                        dataKey="y"
                        stroke={getModelColor(modelId)}
                        strokeWidth={2}
                        dot={false}
                        name={`${modelId} Pareto`}
                        isAnimationActive={false}
                      />
                    );
                  })
              ) : (
                // Global Pareto frontier line
                <Line
                  data={paretoLineData}
                  dataKey="y"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Pareto Frontier"
                  isAnimationActive={false}
                />
              )}

              {/* Scatter points - colored by model or by Pareto status */}
              {colorByModel ? (
                // Per-model colored scatter points - suboptimal first, then Pareto on top
                modelIds.flatMap(modelId => {
                  const modelData = chartData.filter(d => {
                    const dModelId = 'model_id' in d.config ? (d.config as Configuration).model_id : '';
                    return dModelId === modelId;
                  });
                  const isGrayed = hoveredLegendModel !== null && hoveredLegendModel !== modelId;
                  const color = isGrayed ? '#6b7280' : getModelColor(modelId);
                  const suboptimalData = modelData.filter(d => !d.isPareto);
                  const paretoData = modelData.filter(d => d.isPareto);
                  return [
                    <Scatter
                      key={`scatter-suboptimal-${modelId}`}
                      data={suboptimalData}
                      fill={color}
                      fillOpacity={isGrayed ? 0.15 : 0.5}
                      stroke={color}
                      strokeOpacity={isGrayed ? 0.15 : 0.7}
                      strokeWidth={1}
                      isAnimationActive={false}
                    />,
                    <Scatter
                      key={`scatter-pareto-${modelId}`}
                      data={paretoData}
                      fill={color}
                      fillOpacity={isGrayed ? 0.15 : 0.85}
                      stroke={color}
                      strokeOpacity={isGrayed ? 0.15 : 1}
                      strokeWidth={1}
                      isAnimationActive={false}
                    />,
                  ];
                })
              ) : (
                <>
                  {/* All points (non-Pareto) */}
                  <Scatter
                    data={chartData.filter(d => !d.isPareto)}
                    fill="#94a3b8"
                    fillOpacity={0.5}
                    isAnimationActive={false}
                  />

                  {/* Pareto points (highlighted) */}
                  <Scatter
                    data={chartData.filter(d => d.isPareto)}
                    fill="#f59e0b"
                    stroke="#d97706"
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>

          {/* Custom floating tooltip */}
          {activeData && mousePosition && (
            <FloatingTooltip
              data={activeData}
              mouseX={mousePosition.x}
              mouseY={mousePosition.y}
            />
          )}
        </div>
      )}

      {/* Legend */}
      {colorByModel ? (
        // Per-model legend with hover highlighting and click to open detail
        <div className="flex flex-wrap items-center justify-center gap-4 mt-2 text-sm text-gray-600 dark:text-gray-400">
          {modelIds.map(modelId => {
            const color = getModelColor(modelId);
            const nickname = modelNicknames.get(modelId) || modelId;
            const isGrayed = hoveredLegendModel !== null && hoveredLegendModel !== modelId;
            const displayColor = isGrayed ? '#d1d5db' : color;
            return (
              <div
                key={modelId}
                className={`flex items-center gap-2 cursor-pointer ${onLegendClick ? 'hover:underline' : ''}`}
                onMouseEnter={() => setHoveredLegendModel(modelId)}
                onMouseLeave={() => setHoveredLegendModel(null)}
                onClick={() => onLegendClick?.(modelId)}
              >
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: displayColor }}
                />
                <div
                  className="w-4 border-t-2 flex-shrink-0"
                  style={{ borderColor: displayColor }}
                />
                <span className={isGrayed ? 'text-gray-400' : ''}>{nickname}</span>
              </div>
            );
          })}
        </div>
      ) : (
        // Default legend
        <div className="flex items-center justify-center gap-6 mt-2 text-sm text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500 border-2 border-amber-600" />
            <span>Pareto optimal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-slate-400 opacity-50" />
            <span>Suboptimal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 border-t-2 border-amber-500" />
            <span>Pareto frontier</span>
          </div>
        </div>
      )}
    </div>
  );
}
