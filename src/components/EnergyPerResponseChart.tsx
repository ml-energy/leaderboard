import { useMemo } from 'react';
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
  defaultEnergyPerToken: number;
  maxEnergyPerToken: number;
  configLabel?: string;
}

interface ChartDataPoint {
  energyStart: number;
  energyEnd: number;
  energyMidpoint: number;
  count: number;
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
  defaultEnergyPerToken: _defaultEnergyPerToken,
  maxEnergyPerToken,
  configLabel,
}: EnergyPerResponseChartProps) {
  const chartData = useMemo(() => {
    if (!energyPerToken) return null;
    return transformToEnergyDistribution(outputLengthDistribution, energyPerToken);
  }, [outputLengthDistribution, energyPerToken]);

  // Fixed X-axis max based on maximum energy/token across all configs
  const maxOutputLength = outputLengthDistribution.bins[outputLengthDistribution.bins.length - 1];
  const xAxisMax = maxOutputLength * maxEnergyPerToken;

  if (!chartData || !energyPerToken) {
    return (
      <div className="flex items-center justify-center h-full min-h-[350px] bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
        <p className="text-gray-500 dark:text-gray-400 text-sm">
          Hover over a configuration to see the energy/response distribution
        </p>
      </div>
    );
  }

  // Calculate bar width as a fraction of X max for proper positioning
  const barWidthFraction = chartData && chartData.length > 0
    ? (chartData[0].energyEnd - chartData[0].energyStart) / xAxisMax * 100
    : 2;

  return (
    <div className="h-full flex flex-col">
      {configLabel && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
          Config: <span className="font-medium text-gray-900 dark:text-white">{configLabel}</span>
        </p>
      )}
      <div className="flex-1 min-h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
          data={chartData}
          margin={{ bottom: 60, left: 10, right: 30, top: 10 }}
          barCategoryGap={0}
        >
          <CartesianGrid strokeDasharray="3 3" className="opacity-50" />
          <XAxis
            dataKey="energyMidpoint"
            type="number"
            domain={[0, xAxisMax]}
            tick={{ fontSize: 10 }}
            tickFormatter={(value: number) => value.toFixed(0)}
            label={{
              value: 'Energy per Response [J]',
              position: 'insideBottom',
              offset: -10,
              className: 'fill-gray-600 dark:fill-gray-400',
            }}
          />
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
          <Bar
            dataKey="count"
            fill="#10b981"
            barSize={`${barWidthFraction}%`}
          >
            {chartData?.map((_, index) => (
              <Cell key={`cell-${index}`} />
            ))}
          </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
