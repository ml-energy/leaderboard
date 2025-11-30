interface SidebarProps {
  task: string;
  latencyDeadline: number;
  onLatencyDeadlineChange: (value: number) => void;
  defaultLatencyDeadline: number;
  maxLatencyDeadline: number;
  energyBudget: number;
  onEnergyBudgetChange: (value: number) => void;
  maxEnergyBudget: number;
  selectedGPUs: Set<string>;
  onGPUToggle: (gpu: string) => void;
  availableGPUs: string[];
}

export default function Sidebar({
  task,
  latencyDeadline,
  onLatencyDeadlineChange,
  defaultLatencyDeadline,
  maxLatencyDeadline,
  energyBudget,
  onEnergyBudgetChange,
  maxEnergyBudget,
  selectedGPUs,
  onGPUToggle,
  availableGPUs,
}: SidebarProps) {
  // Labels based on task type
  const isDiffusion = task === 'text-to-image' || task === 'text-to-video';
  const latencyLabel = isDiffusion ? 'Batch latency deadline' : 'Median ITL deadline';
  const latencyUnit = isDiffusion ? 's' : 'ms';
  const latencyStep = isDiffusion ? 1 : 10;
  const energyLabel = task === 'text-to-image'
    ? 'Per image energy budget'
    : task === 'text-to-video'
      ? 'Per video energy budget'
      : 'Per token energy budget';

  return (
    <div>
      <div className="flex flex-wrap items-center gap-6">
        <div className="min-w-[250px] max-w-[400px]">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {latencyLabel}: <span className="font-mono text-blue-600 dark:text-blue-400">{isDiffusion ? latencyDeadline.toFixed(1) : latencyDeadline}</span> {latencyUnit}
          </label>
          <input
            type="range"
            min="0"
            max={maxLatencyDeadline}
            step={latencyStep}
            value={latencyDeadline}
            onChange={(e) => onLatencyDeadlineChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        <div className="min-w-[250px] max-w-[400px]">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {energyLabel}: <span className="font-mono text-blue-600 dark:text-blue-400">{isDiffusion ? energyBudget.toFixed(0) : energyBudget.toFixed(2)}</span> J
          </label>
          <input
            type="range"
            min="0"
            max={maxEnergyBudget}
            step={maxEnergyBudget / 100}
            value={energyBudget}
            onChange={(e) => onEnergyBudgetChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GPU Models
          </label>
          <div className="flex flex-wrap gap-2">
            {availableGPUs.map((gpu) => (
              <label
                key={gpu}
                className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 px-3 py-1.5 rounded-md border border-gray-200 dark:border-gray-700"
              >
                <input
                  type="checkbox"
                  checked={selectedGPUs.has(gpu)}
                  onChange={() => onGPUToggle(gpu)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">{gpu}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              onLatencyDeadlineChange(defaultLatencyDeadline);
              onEnergyBudgetChange(maxEnergyBudget);
              availableGPUs.forEach((gpu) => {
                if (!selectedGPUs.has(gpu)) {
                  onGPUToggle(gpu);
                }
              });
            }}
            className="px-4 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
