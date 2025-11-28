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
}

type XAxisScale = 'linear' | 'log';

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

  const chartData = useMemo(() => {
    if (!energyPerToken) return null;
    return transformToEnergyDistribution(outputLengthDistribution, energyPerToken);
  }, [outputLengthDistribution, energyPerToken]);

  if (!chartData || !energyPerToken || !maxEnergyPerResponse) {
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

  // For log scale, use a fixed minimum (10 J) to ensure consistent axis range
  const xAxisMin = xAxisScale === 'log' ? 10 : 0;

  // Calculate bar width as a fraction of X max for proper positioning
  const barWidthFraction = chartData.length > 0
    ? (chartData[0].energyEnd - chartData[0].energyStart) / xAxisMax * 100
    : 2;

  // Generate log scale ticks (powers of 10 and intermediates)
  const generateLogTicks = (min: number, max: number): number[] => {
    const ticks: number[] = [];
    const minLog = Math.floor(Math.log10(min));
    const maxLog = Math.ceil(Math.log10(max));
    for (let i = minLog; i <= maxLog; i++) {
      const base = Math.pow(10, i);
      if (base >= min && base <= max) ticks.push(base);
      // Add intermediate values (2, 5) for better granularity
      if (base * 2 >= min && base * 2 <= max) ticks.push(base * 2);
      if (base * 5 >= min && base * 5 <= max) ticks.push(base * 5);
    }
    return ticks.sort((a, b) => a - b);
  };

  const logTicks = xAxisScale === 'log' ? generateLogTicks(xAxisMin, xAxisMax) : undefined;

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
            data={chartData}
            margin={{ bottom: 60, left: 10, right: 30, top: 10 }}
            barCategoryGap={0}
          >
            <CartesianGrid strokeDasharray="3 3" className="opacity-50" />
            <XAxis
              dataKey="energyMidpoint"
              type="number"
              scale={xAxisScale === 'log' ? 'log' : undefined}
              domain={[xAxisMin, xAxisMax]}
              allowDataOverflow={true}
              ticks={logTicks}
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

      {/* Config label (below chart, like legend) */}
      {configLabel && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 text-center">
          Config: <span className="font-medium text-gray-900 dark:text-white">{configLabel}</span>
        </p>
      )}
    </div>
  );
}
