import { useEffect, useState, useMemo } from 'react';
import { IndexData, TaskData, Configuration } from './types';
import { loadIndexData, loadTaskData } from './utils/dataLoader';
import { DEFAULT_COLUMNS, ADVANCED_COLUMNS } from './config/columns';
import TaskTabs from './components/TaskTabs';
import Sidebar from './components/Sidebar';
import LeaderboardTable from './components/LeaderboardTable';
import { ModelDetailModal } from './components/ModelDetailModal';
import { AboutPage } from './components/AboutPage';

function App() {
  const [indexData, setIndexData] = useState<IndexData | null>(null);
  const [taskData, setTaskData] = useState<TaskData | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingTask, setLoadingTask] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeTask, setActiveTask] = useState<string>('');
  const [latencyDeadline, setLatencyDeadline] = useState<number>(500);
  const [selectedGPUs, setSelectedGPUs] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [selectedConfig, setSelectedConfig] = useState<Configuration | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);

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

    setLoadingTask(true);
    loadTaskData(activeTask)
      .then((data) => {
        setTaskData(data);
        setLoadingTask(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoadingTask(false);
      });
  }, [activeTask]);

  const availableGPUs = useMemo(() => {
    if (!taskData) return [];
    const gpuSet = new Set<string>();
    taskData.configurations.forEach((config) => {
      gpuSet.add(config.gpu_model);
    });
    return Array.from(gpuSet).sort();
  }, [taskData]);

  useEffect(() => {
    if (availableGPUs.length > 0 && selectedGPUs.size === 0) {
      setSelectedGPUs(new Set(availableGPUs));
    }
  }, [availableGPUs]);

  const getBestConfigPerModel = (
    configs: Configuration[],
    deadline: number,
    gpus: Set<string>
  ): Configuration[] => {
    const byModel = new Map<string, Configuration[]>();

    configs.forEach((config) => {
      if (!gpus.has(config.gpu_model)) return;
      if (config.median_itl_ms > deadline) return;

      if (!byModel.has(config.model_id)) {
        byModel.set(config.model_id, []);
      }
      byModel.get(config.model_id)!.push(config);
    });

    const bestConfigs: Configuration[] = [];
    byModel.forEach((modelConfigs) => {
      const best = modelConfigs.reduce((prev, curr) =>
        curr.energy_per_token_joules < prev.energy_per_token_joules ? curr : prev
      );
      bestConfigs.push(best);
    });

    return bestConfigs.sort(
      (a, b) => a.energy_per_token_joules - b.energy_per_token_joules
    );
  };

  const filteredConfigs = useMemo(() => {
    if (!taskData) return [];
    return getBestConfigPerModel(
      taskData.configurations,
      latencyDeadline,
      selectedGPUs
    );
  }, [taskData, latencyDeadline, selectedGPUs]);

  const columns = useMemo(() => {
    return showAdvanced ? [...DEFAULT_COLUMNS, ...ADVANCED_COLUMNS] : DEFAULT_COLUMNS;
  }, [showAdvanced]);

  const handleGPUToggle = (gpu: string) => {
    const newSet = new Set(selectedGPUs);
    if (newSet.has(gpu)) {
      newSet.delete(gpu);
    } else {
      newSet.add(gpu);
    }
    setSelectedGPUs(newSet);
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
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="mx-auto py-6 px-6 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              The ML.ENERGY Leaderboard
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              Benchmarking energy efficiency of LLMs and MLLMs
            </p>
          </div>
          <button
            onClick={() => setAboutOpen(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            About
          </button>
        </div>
      </header>

      <main className="mx-auto py-6 px-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow mb-6">
          <div className="px-6 pt-6">
            <TaskTabs
              tasks={indexData?.tasks || []}
              activeTask={activeTask}
              onTaskChange={setActiveTask}
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
                <Sidebar
                  latencyDeadline={latencyDeadline}
                  onLatencyDeadlineChange={setLatencyDeadline}
                  selectedGPUs={selectedGPUs}
                  onGPUToggle={handleGPUToggle}
                  availableGPUs={availableGPUs}
                  showAdvanced={showAdvanced}
                  onShowAdvancedChange={setShowAdvanced}
                />

                <div>
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                        {taskData?.task_display_name || activeTask}
                      </h2>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        Showing {filteredConfigs.length} models meeting criteria
                      </p>
                    </div>
                  </div>

                  <LeaderboardTable
                    configurations={filteredConfigs}
                    columns={columns}
                    onRowClick={(config) => {
                      setSelectedConfig(config);
                      setModalOpen(true);
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="bg-white dark:bg-gray-800 shadow mt-12">
        <div className="mx-auto py-4 px-6">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            Last updated: {indexData?.last_updated || 'Unknown'}
          </p>
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            Version: 3.0
          </p>
        </div>
      </footer>

      {selectedConfig && (
        <ModelDetailModal
          modelId={selectedConfig.model_id}
          task={activeTask}
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          currentConfig={{
            gpu_model: selectedConfig.gpu_model,
            num_gpus: selectedConfig.num_gpus,
            max_num_seqs: selectedConfig.max_num_seqs,
          }}
        />
      )}

      {aboutOpen && <AboutPage onClose={() => setAboutOpen(false)} />}
    </div>
  );
}

export default App;
