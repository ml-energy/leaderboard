import { getTaskConfig } from '../config/tasks';
import type { Architecture } from '../types';

interface TaskTabsProps {
  tasks: string[];
  activeTask: string;
  onTaskChange: (task: string) => void;
  architectures?: Record<Architecture, string[]>;
}

const ARCHITECTURE_LABELS: Record<Architecture, string> = {
  llm: 'LLM',
  mllm: 'MLLM',
  diffusion: 'Diffusion',
};

export default function TaskTabs({ tasks, activeTask, onTaskChange, architectures }: TaskTabsProps) {
  const getTabLabel = (task: string) => {
    return getTaskConfig(task).tabLabel;
  };

  // If architectures is provided, group tasks by architecture
  const groupedTasks: { architecture: Architecture; tasks: string[] }[] = [];
  if (architectures) {
    const archOrder: Architecture[] = ['llm', 'mllm', 'diffusion'];
    for (const arch of archOrder) {
      const archTasks = architectures[arch]?.filter(t => tasks.includes(t)) || [];
      if (archTasks.length > 0) {
        groupedTasks.push({ architecture: arch, tasks: archTasks });
      }
    }
  }

  // If no grouping, show flat list
  if (groupedTasks.length === 0) {
    return (
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8 overflow-x-auto" aria-label="Tabs">
          {tasks.map((task) => (
            <button
              key={task}
              onClick={() => onTaskChange(task)}
              className={`
                whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
                ${
                  activeTask === task
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
              aria-current={activeTask === task ? 'page' : undefined}
            >
              {getTabLabel(task)}
            </button>
          ))}
        </nav>
      </div>
    );
  }

  // Grouped tabs with architecture labels
  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <nav className="-mb-px flex items-center overflow-x-auto" aria-label="Tabs">
        {groupedTasks.map((group, groupIndex) => (
          <div key={group.architecture} className="flex items-center">
            {/* Architecture label */}
            <span className="text-sm font-semibold text-gray-600 dark:text-gray-300 tracking-wide px-2 py-4">
              {ARCHITECTURE_LABELS[group.architecture]}
            </span>
            {/* Tasks in this architecture */}
            <div className="flex space-x-4">
              {group.tasks.map((task) => (
                <button
                  key={task}
                  onClick={() => onTaskChange(task)}
                  className={`
                    whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
                    ${
                      activeTask === task
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                    }
                  `}
                  aria-current={activeTask === task ? 'page' : undefined}
                >
                  {getTabLabel(task)}
                </button>
              ))}
            </div>
            {/* Separator between groups */}
            {groupIndex < groupedTasks.length - 1 && (
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-600 mx-4" />
            )}
          </div>
        ))}
      </nav>
    </div>
  );
}
