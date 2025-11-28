import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  TooltipProps,
} from 'recharts';
import { Distribution } from '../types';

interface EnergyPerResponseChartProps {
  outputLengthDistribution: Distribution;
  energyPerToken: number | null;
  maxEnergyPerResponse: number | null;  // Fixed X-axis max: actual max energy across all configs (null if not yet calculated)
  configLabel?: string;
}

interface ChartDataPoint {
  energyStart: number;
  energyEnd: number;
  energyMidpoint: number;
  count: number;
  // For log scale: index position for even spacing
  logIndex?: number;
}

type XAxisScale = 'linear' | 'log';

// Generate log-scale bin boundaries with finer granularity
// Uses 1, 1.5, 2, 3, 5, 7 sequence (6 bins per decade instead of 3)
function generate125Bins(min: number, max: number): number[] {
  const bins: number[] = [];
  const multipliers = [1, 1.5, 2, 3, 5, 7];
  let power = Math.floor(Math.log10(min));

  while (bins.length === 0 || bins[bins.length - 1] < max * 2) {
    for (const m of multipliers) {
      const value = m * Math.pow(10, power);
      if (value >= min * 0.5 && value <= max * 2) {
        bins.push(value);
      }
    }
    power++;
    if (power > 10) break; // Safety limit
  }
  return bins;
}

// Rebin data for log scale using 1-2-5 sequence
// Returns data with logIndex for even bar spacing
function rebinToLogScale(
  linearData: ChartDataPoint[],
  minEnergy: number,
  maxEnergy: number
): ChartDataPoint[] {
  const logBins = generate125Bins(minEnergy, maxEnergy);

  const result: ChartDataPoint[] = [];
  for (let i = 0; i < logBins.length - 1; i++) {
    const binStart = logBins[i];
    const binEnd = logBins[i + 1];
    const binMidpoint = Math.sqrt(binStart * binEnd); // Geometric mean

    // Sum counts from linear bins whose midpoints fall in this range
    let count = 0;
    for (const d of linearData) {
      if (d.energyMidpoint >= binStart && d.energyMidpoint < binEnd) {
        count += d.count;
      }
    }

    // Add logIndex for positioning on evenly-spaced axis
    result.push({
      energyStart: binStart,
      energyEnd: binEnd,
      energyMidpoint: binMidpoint,
      count,
      logIndex: i,
    });
  }

  return result; // Keep all bins (even empty ones) for proper spacing
}

function transformToEnergyDistribution(
  distribution: Distribution,
  energyPerToken: number
): ChartDataPoint[] {
  const { bins, counts } = distribution;
  return counts.map((count, i) => {
    const binStart = bins[i];
    const binEnd = bins[i + 1];
    const energyStart = binStart * energyPerToken;
    const energyEnd = binEnd * energyPerToken;
    const energyMidpoint = (energyStart + energyEnd) / 2;
    return {
      energyStart,
      energyEnd,
      energyMidpoint,
      count,
    };
  });
}

function CustomBarTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload as ChartDataPoint;
  return (
    <div className="bg-white dark:bg-gray-800 p-2 rounded shadow border border-gray-200 dark:border-gray-700 text-sm">
      <p className="text-gray-900 dark:text-white">
        Energy: {data.energyStart.toFixed(0)}-{data.energyEnd.toFixed(0)} J
      </p>
      <p className="text-gray-600 dark:text-gray-300">
        Requests: {data.count}
      </p>
    </div>
  );
}

export function EnergyPerResponseChart({
  outputLengthDistribution,
  energyPerToken,
  maxEnergyPerResponse,
  configLabel,
}: EnergyPerResponseChartProps) {
  const [xAxisScale, setXAxisScale] = useState<XAxisScale>('linear');

  // Linear scale: use original bins
  const linearChartData = useMemo(() => {
    if (!energyPerToken) return null;
    return transformToEnergyDistribution(outputLengthDistribution, energyPerToken);
  }, [outputLengthDistribution, energyPerToken]);

  // Log scale: rebin into 1-2-5 sequence for equal-width bars
  const logChartData = useMemo(() => {
    if (!linearChartData || !maxEnergyPerResponse) return null;
    return rebinToLogScale(linearChartData, 10, maxEnergyPerResponse);
  }, [linearChartData, maxEnergyPerResponse]);

  // Select data based on current scale
  const chartData = xAxisScale === 'log' ? logChartData : linearChartData;

  if (!linearChartData || !energyPerToken || !maxEnergyPerResponse) {
    return (
      <div className="flex items-center justify-center h-full min-h-[350px] bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
        <p className="text-gray-500 dark:text-gray-400 text-sm">
          Hover over a configuration to see the energy/response distribution
        </p>
      </div>
    );
  }

  // After the null check, maxEnergyPerResponse is guaranteed to be a number
  const xAxisMax = maxEnergyPerResponse;

  // For linear scale: calculate bar width as fraction of X range
  const linearBarWidthFraction = chartData && chartData.length > 0 && xAxisScale === 'linear'
    ? (chartData[0].energyEnd - chartData[0].energyStart) / xAxisMax * 100
    : 2;

  // For log scale: format tick label showing bin range
  const formatLogTick = (logIndex: number): string => {
    if (!logChartData || logIndex < 0 || logIndex >= logChartData.length) return '';
    const d = logChartData[logIndex];
    // Show bin start value (e.g., "100" for bin 100-200)
    return d.energyStart.toFixed(0);
  };

  const scaleOptions: { value: XAxisScale; label: string }[] = [
    { value: 'linear', label: 'Linear' },
    { value: 'log', label: 'Log' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Scale selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {scaleOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => setXAxisScale(opt.value)}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${
              xAxisScale === opt.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData ?? undefined}
            margin={{ bottom: 60, left: 10, right: 30, top: 10 }}
            barCategoryGap={0}
          >
            <CartesianGrid strokeDasharray="3 3" className="opacity-50" />
            {xAxisScale === 'linear' ? (
              <XAxis
                dataKey="energyMidpoint"
                type="number"
                domain={[0, xAxisMax]}
                allowDataOverflow={true}
                tick={{ fontSize: 10 }}
                tickFormatter={(value: number) => value.toFixed(0)}
                label={{
                  value: 'Energy per Response [J]',
                  position: 'insideBottom',
                  offset: -10,
                  className: 'fill-gray-600 dark:fill-gray-400',
                }}
              />
            ) : (
              <XAxis
                dataKey="logIndex"
                type="number"
                domain={logChartData ? [-0.5, logChartData.length - 0.5] : [0, 1]}
                allowDataOverflow={true}
                ticks={logChartData?.map((_, i) => i)}
                tick={{ fontSize: 10 }}
                tickFormatter={formatLogTick}
                label={{
                  value: 'Energy per Response [J] (log scale)',
                  position: 'insideBottom',
                  offset: -10,
                  className: 'fill-gray-600 dark:fill-gray-400',
                }}
              />
            )}
            <YAxis
              tick={{ fontSize: 12 }}
              label={{
                value: 'Count',
                angle: -90,
                position: 'insideLeft',
                className: 'fill-gray-600 dark:fill-gray-400',
              }}
            />
            <Tooltip content={<CustomBarTooltip />} />
            {xAxisScale === 'linear' ? (
              <Bar
                dataKey="count"
                fill="#10b981"
                barSize={`${linearBarWidthFraction}%`}
              >
                {chartData?.map((_, index) => (
                  <Cell key={`cell-${index}`} />
                ))}
              </Bar>
            ) : (
              <Bar
                dataKey="count"
                fill="#10b981"
              >
                {chartData?.map((_, index) => (
                  <Cell key={`cell-${index}`} />
                ))}
              </Bar>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Config label (below chart, like legend) */}
      {configLabel && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 text-center">
          Config: <span className="font-medium text-gray-900 dark:text-white">{configLabel}</span>
        </p>
      )}
    </div>
  );
}
