import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { IndexData, TaskData, AnyConfiguration, Configuration } from './types';
import { loadIndexData, loadTaskData } from './utils/dataLoader';
import { getColumnsForTask } from './config/columns';
import { getTaskConfig, sortTasks, isDiffusionTask } from './config/tasks';
import TaskTabs from './components/TaskTabs';
import Sidebar from './components/Sidebar';
import LeaderboardTable from './components/LeaderboardTable';
import { ModelDetailModal } from './components/ModelDetailModal';
import { ComparisonModal } from './components/ComparisonModal';
import { AboutPage } from './components/AboutPage';
import { TaskAboutModal } from './components/TaskAboutModal';
import { TimeEnergyTradeoffChart, ITLPercentile } from './components/TimeEnergyTradeoffChart';
import { AnnouncementBanner } from './components/AnnouncementBanner';
import { AnnouncementsModal } from './components/AnnouncementsModal';
import { announcements } from './config/announcements';

function App() {
  const [indexData, setIndexData] = useState<IndexData | null>(null);
  const [taskData, setTaskData] = useState<TaskData | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingTask, setLoadingTask] = useState(false);
  const isTaskSwitchingRef = useRef(false); // Ref for synchronous check in effects
  const [error, setError] = useState<string | null>(null);

  const [activeTask, setActiveTask] = useState<string>('');
  const [latencyDeadline, setLatencyDeadline] = useState<number>(500); // ITL (ms) for LLM/MLLM, batch latency (s) for diffusion
  const [energyBudget, setEnergyBudget] = useState<number | null>(null);
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [selectedConfig, setSelectedConfig] = useState<AnyConfiguration | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [taskAboutOpen, setTaskAboutOpen] = useState(false);
  const [announcementsOpen, setAnnouncementsOpen] = useState(false);
  const [readAnnouncementIds, setReadAnnouncementIds] = useState<Set<string>>(() => {
    const stored = localStorage.getItem('readAnnouncementIds');
    if (stored) {
      try {
        return new Set(JSON.parse(stored));
      } catch {
        return new Set();
      }
    }
    return new Set();
  });

  // Multi-model comparison state (single source of truth: selectedForCompare)
  const [selectedForCompare, setSelectedForCompare] = useState<Set<string>>(new Set());
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
  const [selectedPercentile, setSelectedPercentile] = useState<ITLPercentile>('p50');
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const stored = localStorage.getItem('darkMode');
    if (stored !== null) return stored === 'true';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', String(darkMode));
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('readAnnouncementIds', JSON.stringify([...readAnnouncementIds]));
  }, [readAnnouncementIds]);

  const unreadAnnouncementCount = announcements.filter(a => !readAnnouncementIds.has(a.date)).length;

  const handleMarkAnnouncementAsRead = (id: string) => {
    setReadAnnouncementIds(prev => new Set([...prev, id]));
  };

  const handleMarkAllAnnouncementsAsRead = useCallback(() => {
    setReadAnnouncementIds(new Set(announcements.map(a => a.date)));
  }, []);

  useEffect(() => {
    loadIndexData()
      .then((data) => {
        setIndexData(data);
        if (data.tasks.length > 0) {
          setActiveTask(data.tasks[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!activeTask) return;

    // Set ref synchronously so other effects in this cycle can check it
    isTaskSwitchingRef.current = true;
    setLoadingTask(true);
    loadTaskData(activeTask)
      .then((data) => {
        setTaskData(data);
        setLoadingTask(false);
        isTaskSwitchingRef.current = false;
      })
      .catch((err) => {
        setError(err.message);
        setLoadingTask(false);
        isTaskSwitchingRef.current = false;
      });
  }, [activeTask]);

  useEffect(() => {
    if (!activeTask) return;
    const config = getTaskConfig(activeTask);
    // Set default latency based on task type
    const defaultLatency = isDiffusionTask(activeTask)
      ? config.defaultLatencyDeadlineS ?? 10
      : config.defaultItlDeadlineMs ?? 500;
    setLatencyDeadline(defaultLatency);
    setEnergyBudget(config.defaultEnergyBudgetJ);
  }, [activeTask]);

  const availableGPUs = useMemo(() => {
    if (!taskData) return [];
    const gpuSet = new Set<string>();
    taskData.configurations.forEach((config) => {
      gpuSet.add(config.gpu_model);
    });
    return Array.from(gpuSet).sort();
  }, [taskData]);

  // Helper to get energy value from any configuration type
  const getEnergyValue = (config: AnyConfiguration): number => {
    if ('energy_per_image_joules' in config) return config.energy_per_image_joules;
    if ('energy_per_video_joules' in config) return config.energy_per_video_joules;
    return (config as Configuration).energy_per_token_joules;
  };

  // Helper to get latency value from any configuration type
  const getLatencyValue = (config: AnyConfiguration): number => {
    if ('batch_latency_s' in config) return config.batch_latency_s;
    return (config as Configuration).median_itl_ms;
  };

  const maxEnergyBudget = useMemo(() => {
    if (!taskData || taskData.configurations.length === 0) return 0.1;
    return Math.max(...taskData.configurations.map(getEnergyValue));
  }, [taskData]);

  const maxLatencyDeadline = useMemo(() => {
    if (!taskData || taskData.configurations.length === 0) {
      return isDiffusionTask(activeTask) ? 120 : 500;
    }
    const maxLatency = Math.max(...taskData.configurations.map(getLatencyValue));
    if (isDiffusionTask(activeTask)) {
      // For diffusion: round up to nice value (10s increments)
      return Math.ceil(maxLatency / 10) * 10;
    }
    // For LLM/MLLM: smallest multiple of 100 larger than maxITL
    return Math.ceil(maxLatency / 100) * 100;
  }, [taskData, activeTask]);

  // Clamp latencyDeadline to maxLatencyDeadline when it changes
  // Skip while task is switching to avoid race with default-setting effect
  useEffect(() => {
    if (isTaskSwitchingRef.current) return;
    if (latencyDeadline > maxLatencyDeadline) {
      setLatencyDeadline(maxLatencyDeadline);
    }
  }, [maxLatencyDeadline, latencyDeadline]);

  useEffect(() => {
    if (availableGPUs.length > 0) {
      setSelectedGPUs(new Set(availableGPUs));
    }
  }, [availableGPUs]);

  useEffect(() => {
    // Skip while loading to avoid using stale taskData from previous task
    if (loadingTask) return;
    // Only set energy budget when we have real task data (not fallback 0.1)
    if (!taskData || taskData.configurations.length === 0) return;
    if (energyBudget === null) {
      setEnergyBudget(maxEnergyBudget);
    }
  }, [maxEnergyBudget, taskData, energyBudget, loadingTask]);

  // Clear comparison selection when switching tasks
  useEffect(() => {
    setSelectedForCompare(new Set());
    setComparisonModalOpen(false);
  }, [activeTask]);

  const getBestConfigPerModel = (
    configs: AnyConfiguration[],
    deadline: number,
    budget: number,
    gpus: Set<string>
  ): AnyConfiguration[] => {
    const byModel = new Map<string, AnyConfiguration[]>();

    configs.forEach((config) => {
      if (!gpus.has(config.gpu_model)) return;
      if (getLatencyValue(config) > deadline) return;
      if (getEnergyValue(config) > budget) return;

      if (!byModel.has(config.model_id)) {
        byModel.set(config.model_id, []);
      }
      byModel.get(config.model_id)!.push(config);
    });

    const bestConfigs: AnyConfiguration[] = [];
    byModel.forEach((modelConfigs) => {
      const best = modelConfigs.reduce((prev, curr) =>
        getEnergyValue(curr) < getEnergyValue(prev) ? curr : prev
      );
      bestConfigs.push(best);
    });

    return bestConfigs.sort(
      (a, b) => getEnergyValue(a) - getEnergyValue(b)
    );
  };

  const effectiveEnergyBudget = energyBudget ?? maxEnergyBudget;

  const filteredConfigs = useMemo(() => {
    if (!taskData) return [];
    return getBestConfigPerModel(
      taskData.configurations,
      latencyDeadline,
      effectiveEnergyBudget,
      selectedGPUs
    );
  }, [taskData, latencyDeadline, effectiveEnergyBudget, selectedGPUs]);

  // All configs for the chart view (filtered by GPU, latency deadline, and energy budget, not by best-per-model)
  const allFilteredConfigs = useMemo(() => {
    if (!taskData) return [];
    return taskData.configurations.filter(
      (config) =>
        selectedGPUs.has(config.gpu_model) &&
        getLatencyValue(config) <= latencyDeadline &&
        getEnergyValue(config) <= effectiveEnergyBudget
    );
  }, [taskData, selectedGPUs, latencyDeadline, effectiveEnergyBudget]);

  const columns = useMemo(() => {
    const { default: defaultCols, advanced: advancedCols } = getColumnsForTask(activeTask);
    return showAdvanced ? [...defaultCols, ...advancedCols] : defaultCols;
  }, [activeTask, showAdvanced]);

  // Default sort key depends on task type
  const defaultSortKey = useMemo(() => {
    if (activeTask === 'text-to-image') return 'energy_per_image_joules';
    if (activeTask === 'text-to-video') return 'energy_per_video_joules';
    return 'energy_per_token_joules';
  }, [activeTask]);

  const handleGPUToggle = (gpu: string) => {
    const newSet = new Set(selectedGPUs);
    if (newSet.has(gpu)) {
      newSet.delete(gpu);
    } else {
      newSet.add(gpu);
    }
    setSelectedGPUs(newSet);
  };

  // Handle row click - always open model detail modal
  const handleRowClick = (config: AnyConfiguration) => {
    setSelectedConfig(config);
    setModalOpen(true);
  };

  // Add model to comparison from model detail modal
  const handleAddToComparison = (modelId: string) => {
    const newSet = new Set(selectedForCompare);
    newSet.add(modelId);
    setSelectedForCompare(newSet);
    setModalOpen(false);
  };

  // Remove model from comparison (syncs back to checkboxes)
  const handleRemoveFromComparison = (modelId: string) => {
    const newSet = new Set(selectedForCompare);
    newSet.delete(modelId);
    setSelectedForCompare(newSet);
    if (newSet.size < 2) {
      setComparisonModalOpen(false);
    }
  };

  // Open model detail from legend click (by modelId)
  const handleLegendClick = (modelId: string) => {
    const config = allFilteredConfigs.find(c => c.model_id === modelId);
    if (config) {
      setSelectedConfig(config);
      setModalOpen(true);
    }
  };

  // Handle checkbox selection for comparison
  const handleSelectionChange = (modelId: string, selected: boolean) => {
    const newSet = new Set(selectedForCompare);
    if (selected) {
      newSet.add(modelId);
    } else {
      newSet.delete(modelId);
    }
    setSelectedForCompare(newSet);
  };

  // Open comparison modal from checkbox selection
  const handleCompareSelected = () => {
    if (selectedForCompare.size >= 2) {
      setComparisonModalOpen(true);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-700 dark:text-gray-300">Loading leaderboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center text-red-600">
          <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <AnnouncementBanner
        readIds={readAnnouncementIds}
        onDismiss={handleMarkAnnouncementAsRead}
      />
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="mx-auto py-6 px-6 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white" style={{ fontFamily: 'Montserrat, sans-serif' }}>
              The ML.ENERGY Leaderboard
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              How much time and energy do generative AI models consume?
            </p>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              Version 3.0 / Last updated: {__LAST_UPDATED__}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? (
                <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              )}
            </button>
            <button
              onClick={() => setAnnouncementsOpen(true)}
              className="relative p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Announcements"
            >
              <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5.882V19.24a1.76 1.76 0 01-3.417.592l-2.147-6.15M18 13a3 3 0 100-6M5.436 13.683A4.001 4.001 0 017 6h1.832c4.1 0 7.625-1.234 9.168-3v14c-1.543-1.766-5.067-3-9.168-3H7a3.988 3.988 0 01-1.564-.317z" />
              </svg>
              {unreadAnnouncementCount > 0 && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full" />
              )}
            </button>
            <button
              onClick={() => setAboutOpen(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              About
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto py-6 px-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow mb-6">
          <div className="px-6 pt-6">
            <TaskTabs
              tasks={sortTasks(indexData?.tasks || [])}
              activeTask={activeTask}
              onTaskChange={setActiveTask}
              architectures={indexData?.architectures}
            />
          </div>

          <div className="p-6">
            {loadingTask ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-4 text-gray-700 dark:text-gray-300">Loading task data...</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Task title */}
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                    {getTaskConfig(activeTask).displayName}
                  </h2>
                  <button
                    onClick={() => setTaskAboutOpen(true)}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>About</span>
                  </button>
                </div>

                {/* Controls */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 w-fit">
                  <Sidebar
                    task={activeTask}
                    latencyDeadline={latencyDeadline}
                    onLatencyDeadlineChange={setLatencyDeadline}
                    defaultLatencyDeadline={
                      isDiffusionTask(activeTask)
                        ? getTaskConfig(activeTask).defaultLatencyDeadlineS ?? 10
                        : getTaskConfig(activeTask).defaultItlDeadlineMs ?? 500
                    }
                    maxLatencyDeadline={maxLatencyDeadline}
                    energyBudget={effectiveEnergyBudget}
                    onEnergyBudgetChange={setEnergyBudget}
                    maxEnergyBudget={maxEnergyBudget}
                    selectedGPUs={selectedGPUs}
                    onGPUToggle={handleGPUToggle}
                    availableGPUs={availableGPUs}
                  />
                </div>

                <div>
                  {/* All Configurations Chart */}
                  <div className="mb-8">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                      All configurations
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {allFilteredConfigs.length} configurations
                    </p>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <TimeEnergyTradeoffChart
                        configurations={allFilteredConfigs}
                        task={activeTask}
                        selectedPercentile={selectedPercentile}
                        onPercentileChange={setSelectedPercentile}
                        onHoverConfig={() => {}}
                        colorByModel={true}
                        onLegendClick={handleLegendClick}
                        showParetoLine={false}
                      />
                    </div>
                  </div>

                  {/* Leaderboard Table */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                          Energy-optimal points for each model
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {filteredConfigs.length} models satisfy the given constraints (click row for model details).
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <label className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 px-3 py-1.5 rounded-md">
                          <input
                            type="checkbox"
                            checked={showAdvanced}
                            onChange={(e) => setShowAdvanced(e.target.checked)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                          />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Advanced columns
                          </span>
                        </label>
                      </div>
                    </div>

                    <LeaderboardTable
                      configurations={filteredConfigs}
                      columns={columns}
                      defaultSortKey={defaultSortKey}
                      onRowClick={handleRowClick}
                      selectedForCompare={selectedForCompare}
                      onSelectionChange={handleSelectionChange}
                      onCompareClick={handleCompareSelected}
                      onClearSelection={() => setSelectedForCompare(new Set())}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="bg-white dark:bg-gray-800 shadow mt-12">
        <div className="mx-auto py-4 px-6">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            Apache 2.0 · <a href="https://github.com/ml-energy/leaderboard" target="_blank" rel="noopener noreferrer" className="hover:underline">GitHub</a>
          </p>
        </div>
      </footer>

      {selectedConfig && (
        <ModelDetailModal
          modelId={selectedConfig.model_id}
          task={activeTask}
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          onAddToComparison={handleAddToComparison}
          currentConfig={{
            gpu_model: selectedConfig.gpu_model,
            num_gpus: selectedConfig.num_gpus,
            batch_size: 'batch_size' in selectedConfig ? selectedConfig.batch_size : undefined,
            max_num_seqs: 'max_num_seqs' in selectedConfig ? (selectedConfig as Configuration).max_num_seqs : undefined,
          }}
        />
      )}

      {comparisonModalOpen && selectedForCompare.size >= 2 && (
        <ComparisonModal
          modelIds={Array.from(selectedForCompare)}
          task={activeTask}
          isOpen={comparisonModalOpen}
          onClose={() => setComparisonModalOpen(false)}
          onRemoveModel={handleRemoveFromComparison}
          modelNicknames={
            indexData
              ? Object.fromEntries(
                  Object.entries(indexData.models).map(([id, info]) => [id, info.nickname])
                )
              : {}
          }
        />
      )}

      {aboutOpen && <AboutPage onClose={() => setAboutOpen(false)} />}

      <AnnouncementsModal
        isOpen={announcementsOpen}
        onClose={() => {
          setAnnouncementsOpen(false);
          handleMarkAllAnnouncementsAsRead();
        }}
      />

      {taskAboutOpen && (
        <TaskAboutModal
          taskId={activeTask}
          onClose={() => setTaskAboutOpen(false)}
        />
      )}
    </div>
  );
}

export default App;
